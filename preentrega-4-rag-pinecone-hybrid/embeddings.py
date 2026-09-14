"""
embeddings.py
=============

Punto único de configuración del modelo de embeddings usado para INDEXAR
(en `ingest.py`) y para CONSULTAR (en `rag_system.py`). Ambos importan
`get_embeddings()` desde acá — nunca instancian su propio modelo por
separado — para que sea imposible que terminen usando modelos distintos
(el "error #1" de cualquier sistema RAG).

Soporta dos proveedores vía `EMBEDDINGS_PROVIDER` (ver `config.py`):

    - "huggingface" (default): sentence-transformers corriendo LOCAL,
      gratis, sin API key. Ideal para probar todo el pipeline sin costo.
    - "openai": `text-embedding-3-small` vía API, de pago (aunque muy
      barato), mejor calidad semántica para producción real.

Sea cual sea el proveedor, la dimensión del vector resultante tiene que
coincidir con `config.EMBEDDING_DIMENSION` (usada por `pinecone_setup.py`
para crear el índice) — por eso `config.py` ya ajusta el default de esa
dimensión según el proveedor elegido.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from config import EMBEDDING_MODEL, EMBEDDINGS_PROVIDER, HUGGINGFACE_EMBEDDING_MODEL

_embeddings_cache: Embeddings | None = None


def get_embeddings() -> Embeddings:
    global _embeddings_cache
    if _embeddings_cache is not None:
        return _embeddings_cache

    if EMBEDDINGS_PROVIDER == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        _embeddings_cache = HuggingFaceEmbeddings(model_name=HUGGINGFACE_EMBEDDING_MODEL)

    elif EMBEDDINGS_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        _embeddings_cache = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    else:
        raise ValueError(
            f"EMBEDDINGS_PROVIDER desconocido: '{EMBEDDINGS_PROVIDER}'. Usá "
            f"'huggingface' u 'openai'."
        )

    return _embeddings_cache
