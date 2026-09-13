"""
rag_system.py
=============

Recuperador Híbrido (Hybrid Retriever).

`RAGSystem` encapsula un `EnsembleRetriever` de LangChain que combina:
    - Un retriever **léxico** (`BM25Retriever`), fuerte en términos técnicos
      exactos y nombres propios (ej. "ConfigDict", "APIRouter") que un
      embedding puede diluir semánticamente.
    - Un retriever **semántico** (Pinecone vía `PineconeVectorStore`),
      fuerte en similitud de significado aunque no se use la palabra exacta.

Los retrievers son inyectables por constructor (`vector_retriever`,
`bm25_retriever`): en producción se arman solos contra Pinecone y el
corpus local cacheado, pero en los tests se les puede pasar un retriever
falso para verificar la lógica de ensamblado y de top-k sin tocar
Pinecone ni gastar llamadas a la API de embeddings.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config import BM25_WEIGHT, CORPUS_CACHE_PATH, PINECONE_NAMESPACE, RAG_TOP_K, VECTOR_WEIGHT

logger = logging.getLogger("rag_system")


def _importar_ensemble_retriever():
    """`EnsembleRetriever` vivió históricamente en `langchain.retrievers`
    (LangChain <1.0). A partir de LangChain 1.x se reubicó en el paquete
    `langchain_classic.retrievers`. Se intenta primero la ruta clásica y,
    si no existe, se usa la nueva — así el proyecto funciona con
    cualquiera de las dos series de versiones sin tocar código."""
    try:
        from langchain.retrievers import EnsembleRetriever
    except ImportError:
        from langchain_classic.retrievers import EnsembleRetriever
    return EnsembleRetriever


class RAGSystem:
    """Encapsula el EnsembleRetriever (BM25 + Pinecone) del sistema RAG."""

    def __init__(
        self,
        top_k: int = RAG_TOP_K,
        bm25_weight: float = BM25_WEIGHT,
        vector_weight: float = VECTOR_WEIGHT,
        vector_retriever: Optional[BaseRetriever] = None,
        bm25_retriever: Optional[BaseRetriever] = None,
    ) -> None:
        self.top_k = top_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self._vector_retriever_override = vector_retriever
        self._bm25_retriever_override = bm25_retriever
        self._ensemble = None  # se arma lazy, la primera vez que hace falta

    # -- construcción lazy de cada rama (así importar el módulo no exige
    # tener PINECONE_API_KEY/OPENAI_API_KEY configuradas) --------------
    def _construir_vector_retriever(self) -> BaseRetriever:
        from langchain_pinecone import PineconeVectorStore

        from embeddings import get_embeddings
        from pinecone_setup import asegurar_indice

        indice_pinecone = asegurar_indice()
        vectorstore = PineconeVectorStore(
            index=indice_pinecone,
            embedding=get_embeddings(),
            namespace=PINECONE_NAMESPACE,
            text_key="text",
        )
        return vectorstore.as_retriever(search_kwargs={"k": self.top_k})

    def _construir_bm25_retriever(self) -> BaseRetriever:
        from langchain_community.retrievers import BM25Retriever

        from corpus_cache import cargar_corpus

        documentos = cargar_corpus(CORPUS_CACHE_PATH)
        retriever = BM25Retriever.from_documents(documentos)
        retriever.k = self.top_k
        return retriever

    def _get_ensemble(self):
        if self._ensemble is not None:
            return self._ensemble

        EnsembleRetriever = _importar_ensemble_retriever()

        vector_retriever = self._vector_retriever_override or self._construir_vector_retriever()
        bm25_retriever = self._bm25_retriever_override or self._construir_bm25_retriever()

        self._ensemble = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[self.bm25_weight, self.vector_weight],
        )
        logger.info(
            "EnsembleRetriever listo (top_k=%d, pesos bm25=%.2f/vectorial=%.2f).",
            self.top_k,
            self.bm25_weight,
            self.vector_weight,
        )
        return self._ensemble

    # -- API pública -----------------------------------------------------
    def query(self, pregunta: str, k: Optional[int] = None) -> list[Document]:
        """Recupera y combina resultados léxicos + semánticos, devolviendo
        los top-k documentos finales (por defecto, `self.top_k`, entre 3 y
        5 como recomienda el enunciado)."""
        if not pregunta or not pregunta.strip():
            raise ValueError("La consulta no puede estar vacía.")

        ensemble = self._get_ensemble()
        resultados = ensemble.invoke(pregunta)
        k = k or self.top_k
        return resultados[:k]

    async def aquery(self, pregunta: str, k: Optional[int] = None) -> list[Document]:
        """Versión asíncrona de `query()`."""
        if not pregunta or not pregunta.strip():
            raise ValueError("La consulta no puede estar vacía.")

        ensemble = self._get_ensemble()
        resultados = await ensemble.ainvoke(pregunta)
        k = k or self.top_k
        return resultados[:k]
