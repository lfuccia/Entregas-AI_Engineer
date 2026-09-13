"""
tests/test_rag_system.py

Tests unitarios offline para `RAGSystem` / `EnsembleRetriever`. En vez de
Pinecone real, se inyecta un retriever vectorial FALSO (subclase mínima de
BaseRetriever) para verificar que el ensamblado BM25 + vectorial y el
recorte a top-k funcionan, sin gastar llamadas a Pinecone ni a la API de
embeddings de OpenAI.
"""

from __future__ import annotations

from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_system import RAGSystem

CORPUS_DE_PRUEBA = [
    Document(page_content="FastAPI usa APIRouter para organizar módulos grandes.", metadata={"source": "a.md"}),
    Document(page_content="Depends() inyecta dependencias en los endpoints.", metadata={"source": "b.md"}),
    Document(page_content="Pydantic valida el body de la request automáticamente.", metadata={"source": "c.md"}),
    Document(page_content="El middleware se ejecuta antes y después de cada request.", metadata={"source": "d.md"}),
]


class RetrieverVectorialFalso(BaseRetriever):
    """Retriever falso que siempre devuelve un conjunto fijo de documentos,
    simulando lo que devolvería Pinecone sin necesitar credenciales."""

    documentos_a_devolver: List[Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.documentos_a_devolver

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.documentos_a_devolver


def _rag_system_de_prueba(top_k: int = 3) -> RAGSystem:
    bm25 = BM25Retriever.from_documents(CORPUS_DE_PRUEBA)
    bm25.k = top_k

    vectorial_falso = RetrieverVectorialFalso(documentos_a_devolver=CORPUS_DE_PRUEBA[:2])

    return RAGSystem(top_k=top_k, vector_retriever=vectorial_falso, bm25_retriever=bm25)


def test_query_devuelve_documentos_combinados_sin_tocar_pinecone():
    rag = _rag_system_de_prueba(top_k=3)
    resultados = rag.query("¿Cómo se inyectan dependencias en FastAPI?")

    assert isinstance(resultados, list)
    assert all(isinstance(d, Document) for d in resultados)
    assert len(resultados) <= 3


def test_query_respeta_el_top_k_pedido():
    rag = _rag_system_de_prueba(top_k=5)
    resultados = rag.query("middleware", k=2)
    assert len(resultados) <= 2


def test_query_vacia_lanza_excepcion():
    import pytest

    rag = _rag_system_de_prueba()
    with pytest.raises(ValueError):
        rag.query("   ")


def test_ensemble_se_construye_una_sola_vez(monkeypatch):
    """El ensemble se cachea: llamar query() dos veces no debería
    reconstruir el EnsembleRetriever (ni, en el caso real, reabrir la
    conexión a Pinecone) la segunda vez."""
    rag = _rag_system_de_prueba()
    rag.query("primera consulta")
    ensemble_despues_de_la_primera = rag._ensemble

    rag.query("segunda consulta")
    assert rag._ensemble is ensemble_despues_de_la_primera
