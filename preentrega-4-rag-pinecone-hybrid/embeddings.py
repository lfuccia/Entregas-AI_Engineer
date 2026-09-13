"""
embeddings.py
=============

Punto único de configuración del modelo de embeddings usado para INDEXAR
(en `ingest.py`) y para CONSULTAR (en `rag_system.py`). Ambos importan
`get_embeddings()` desde acá — nunca instancian su propio modelo por
separado — para que sea imposible que terminen usando modelos distintos
(el "error #1" de cualquier sistema RAG).

Usa `OpenAIEmbeddings` porque el índice de Pinecone se crea con una
dimensión fija (`config.EMBEDDING_DIMENSION`, default 1536 para
`text-embedding-3-small`): si cambiás de modelo de embeddings, tenés que
actualizar `EMBEDDING_DIMENSION` en el `.env` Y recrear el índice de
Pinecone con esa nueva dimensión (ver `pinecone_setup.py`).
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from config import EMBEDDING_MODEL

_embeddings_cache: Embeddings | None = None


def get_embeddings() -> Embeddings:
    global _embeddings_cache
    if _embeddings_cache is None:
        from langchain_openai import OpenAIEmbeddings

        _embeddings_cache = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return _embeddings_cache
