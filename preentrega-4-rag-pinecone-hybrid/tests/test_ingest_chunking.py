"""
tests/test_ingest_chunking.py

Tests unitarios offline para el chunking de `ingest.py` (mide longitud en
tokens con tiktoken; no requiere Pinecone ni OpenAI).
"""

from __future__ import annotations

from langchain_core.documents import Document

from ingest import _fragmentar


def _documento_largo(source: str, page=None) -> Document:
    contenido = "\n\n".join(
        f"Párrafo {i}: contenido técnico de ejemplo sobre FastAPI y Pydantic. " * 20
        for i in range(20)
    )
    return Document(page_content=contenido, metadata={"source": source, "page": page, "categoria": "test"})


def test_fragmentar_genera_varios_fragmentos_para_documento_largo():
    fragmentos = _fragmentar([_documento_largo("doc_a.md")])
    assert len(fragmentos) > 1


def test_fragmentar_asigna_chunk_index_secuencial_por_fuente():
    fragmentos = _fragmentar([_documento_largo("doc_a.md")])
    indices = [f.metadata["chunk_index"] for f in fragmentos]
    assert indices == list(range(len(fragmentos)))


def test_fragmentar_arma_fuente_id_sin_pagina_para_markdown():
    fragmentos = _fragmentar([_documento_largo("doc_a.md", page=None)])
    assert fragmentos[0].metadata["fuente_id"] == "doc_a.md#fragmento_0"


def test_fragmentar_arma_fuente_id_con_pagina_para_pdf():
    fragmentos = _fragmentar([_documento_largo("doc_b.pdf", page=3)])
    assert fragmentos[0].metadata["fuente_id"] == "doc_b.pdf#p3#fragmento_0"


def test_fragmentar_copia_el_texto_a_la_metadata():
    fragmentos = _fragmentar([_documento_largo("doc_a.md")])
    for fragmento in fragmentos:
        assert fragmento.metadata["text"] == fragmento.page_content


def test_fragmentar_mantiene_contadores_independientes_por_fuente():
    fragmentos = _fragmentar(
        [_documento_largo("doc_a.md"), _documento_largo("doc_b.md")]
    )
    indices_a = [f.metadata["chunk_index"] for f in fragmentos if f.metadata["source"] == "doc_a.md"]
    indices_b = [f.metadata["chunk_index"] for f in fragmentos if f.metadata["source"] == "doc_b.md"]
    assert indices_a == list(range(len(indices_a)))
    assert indices_b == list(range(len(indices_b)))
