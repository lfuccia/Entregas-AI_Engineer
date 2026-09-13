"""
corpus_cache.py
================

BM25 es un recuperador léxico puramente local: necesita el corpus completo
en memoria (no vive "en la nube" como Pinecone). En vez de reconstruirlo
pidiéndole a Pinecone todos sus vectores (lento, y Pinecone no está pensado
para eso), `ingest.py` guarda una copia local de los fragmentos ya
chunkeados en un archivo JSONL (`corpus_cache.jsonl`) en el mismo momento
en que los sube a Pinecone. `rag_system.py` lee ese archivo para construir
el `BM25Retriever`.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document


def guardar_corpus(fragmentos: list[Document], ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8") as f:
        for fragmento in fragmentos:
            f.write(
                json.dumps(
                    {"page_content": fragmento.page_content, "metadata": fragmento.metadata},
                    ensure_ascii=False,
                )
                + "\n"
            )


def cargar_corpus(ruta: Path) -> list[Document]:
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el cache local de corpus '{ruta}'. Corré primero "
            f"'python ingest.py' para generarlo (se crea junto con la "
            f"subida a Pinecone)."
        )

    documentos: list[Document] = []
    with ruta.open("r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            registro = json.loads(linea)
            documentos.append(
                Document(page_content=registro["page_content"], metadata=registro["metadata"])
            )
    return documentos
