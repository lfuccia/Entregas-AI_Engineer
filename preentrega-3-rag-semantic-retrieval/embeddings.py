"""
embeddings.py
=============

Punto único de configuración del modelo de embeddings.

Este módulo existe para eliminar de raíz el "error #1" que menciona el
enunciado: indexar con un modelo de embeddings y consultar con otro. Tanto
`ingest.py` (indexación) como `rag_chain.py` (consulta) importan
`get_embeddings()` desde ACÁ — nunca instancian su propio embeddings model
por separado — así que están garantizados a usar siempre el mismo modelo.

Por default usa un modelo local de HuggingFace (sentence-transformers), que
no requiere API key ni conexión a internet una vez descargado. También
soporta `EMBEDDINGS_PROVIDER=openai` si se prefiere usar la API de OpenAI.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv()


def get_embeddings() -> Embeddings:
    """Devuelve la instancia de embeddings configurada vía
    `EMBEDDINGS_PROVIDER` ('huggingface' por defecto, o 'openai')."""
    provider = os.getenv("EMBEDDINGS_PROVIDER", "huggingface").strip().lower()

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        model_name = os.getenv(
            "HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        return HuggingFaceEmbeddings(model_name=model_name)

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model_name)

    raise ValueError(
        f"EMBEDDINGS_PROVIDER desconocido: '{provider}'. Usá 'huggingface' u "
        f"'openai'."
    )
