"""
ingest.py
=========

Pipeline de Ingesta en Pinecone.

1. Carga documentos (.md, .json, .pdf) desde `./data` con metadatos
   avanzados (fuente, página, categoría) vía `loaders.cargar_documentos`.
2. Los fragmenta con `RecursiveCharacterTextSplitter` (chunking medido en
   TOKENS: 500-800, default 650, con 80 de overlap — el punto medio que
   pide el enunciado entre perder contexto y diluir la precisión del
   embedding).
3. Genera embeddings (OpenAI, ver `embeddings.py`) y los sube a Pinecone
   **incluyendo el texto original en la metadata** (para no depender de
   una base de datos relacional aparte para recuperar el contenido), bajo
   un namespace específico (evita namespace por defecto, que mezclaría
   este dataset con cualquier otro que use el mismo índice).
4. Guarda una copia local del corpus chunkeado (`corpus_cache.jsonl`) para
   que `rag_system.py` pueda construir el `BM25Retriever` sin tener que
   volver a pedirle todos los vectores a Pinecone.

Es idempotente: si el namespace ya tiene vectores, NO vuelve a indexar
(salvo `--force`).

Uso:
    python ingest.py            # indexa solo si el namespace está vacío
    python ingest.py --force    # fuerza re-indexado completo
"""

from __future__ import annotations

import argparse
import logging
import os
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    CORPUS_CACHE_PATH,
    DATA_DIR,
    INDEX_NAME,
    PINECONE_NAMESPACE,
)
from corpus_cache import guardar_corpus
from embeddings import get_embeddings
from loaders import cargar_documentos
from pinecone_setup import asegurar_indice

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag_ingest")


def _fragmentar(documentos: list[Document]) -> list[Document]:
    """RecursiveCharacterTextSplitter midiendo longitud en tokens
    (tiktoken), numerando cada fragmento y armando un 'fuente_id' legible
    (archivo#página#fragmento) para trazabilidad end-to-end."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        separators=["\n## ", "\n\n", "\n", ". ", " ", ""],
    )

    fragmentos: list[Document] = []
    contador_por_fuente: dict[str, int] = {}

    for documento in documentos:
        trozos = splitter.split_documents([documento])
        fuente = documento.metadata["source"]
        for trozo in trozos:
            indice = contador_por_fuente.get(fuente, 0)
            contador_por_fuente[fuente] = indice + 1

            pagina = trozo.metadata.get("page")
            sufijo_pagina = f"#p{pagina}" if pagina is not None else ""
            trozo.metadata["chunk_index"] = indice
            trozo.metadata["fuente_id"] = f"{fuente}{sufijo_pagina}#fragmento_{indice}"
            # El propio texto viaja en la metadata (además de en
            # page_content) para que quede persistido tal cual en Pinecone.
            trozo.metadata["text"] = trozo.page_content
            fragmentos.append(trozo)

        logger.info("  '%s' -> %d fragmento(s).", fuente, contador_por_fuente.get(fuente, 0))

    return fragmentos


def _namespace_ya_poblado(indice_pinecone, namespace: str) -> bool:
    stats = indice_pinecone.describe_index_stats()
    namespaces = stats.get("namespaces", {}) or {}
    cantidad = namespaces.get(namespace, {}).get("vector_count", 0)
    if cantidad:
        logger.info(
            "El namespace '%s' ya tiene %d vectores indexados.", namespace, cantidad
        )
    return cantidad > 0


def ejecutar_ingesta(force: bool = False) -> None:
    from langchain_pinecone import PineconeVectorStore

    indice_pinecone = asegurar_indice()

    if not force and _namespace_ya_poblado(indice_pinecone, PINECONE_NAMESPACE):
        logger.info(
            "Ingesta omitida: el namespace '%s' ya está poblado (usá "
            "--force para re-indexar).",
            PINECONE_NAMESPACE,
        )
        return

    logger.info("Cargando documentos desde '%s'...", DATA_DIR)
    documentos = cargar_documentos(DATA_DIR)
    logger.info("Documentos cargados: %d", len(documentos))

    fragmentos = _fragmentar(documentos)
    logger.info("Total de fragmentos a indexar: %d", len(fragmentos))

    # Copia local para BM25 (y para inspección/debug sin pegarle a Pinecone).
    guardar_corpus(fragmentos, CORPUS_CACHE_PATH)
    logger.info("Corpus local cacheado en '%s'.", CORPUS_CACHE_PATH)

    ids = [str(uuid.uuid4()) for _ in fragmentos]
    vectorstore = PineconeVectorStore(index=indice_pinecone, embedding=get_embeddings())
    vectorstore.add_documents(
        documents=fragmentos,
        ids=ids,
        namespace=PINECONE_NAMESPACE,
    )

    logger.info(
        "Ingesta completa: %d fragmentos subidos a Pinecone (índice='%s', "
        "namespace='%s').",
        len(fragmentos),
        INDEX_NAME,
        PINECONE_NAMESPACE,
    )


if __name__ == "__main__":
    parser_cli = argparse.ArgumentParser(description="Ingesta de documentos a Pinecone.")
    parser_cli.add_argument(
        "--force",
        action="store_true",
        help="Fuerza el re-indexado completo aunque el namespace ya tenga vectores.",
    )
    args = parser_cli.parse_args()
    ejecutar_ingesta(force=args.force)
