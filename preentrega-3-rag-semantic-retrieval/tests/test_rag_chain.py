"""
tests/test_rag_chain.py

Tests unitarios offline (sin LLM, sin embeddings) para el formateo de
contexto de `rag_chain.py`.
"""

from __future__ import annotations

from langchain_core.documents import Document

from rag_chain import _formatear_contexto


def test_formatear_contexto_incluye_etiquetas_de_fuente():
    documentos = [
        Document(
            page_content="Contenido del primer fragmento.",
            metadata={"fuente_id": "doc_a.md#fragmento_0"},
        ),
        Document(
            page_content="Contenido del segundo fragmento.",
            metadata={"fuente_id": "doc_b.md#fragmento_2"},
        ),
    ]

    contexto = _formatear_contexto(documentos)

    assert "[FUENTE: doc_a.md#fragmento_0]" in contexto
    assert "[FUENTE: doc_b.md#fragmento_2]" in contexto
    assert "Contenido del primer fragmento." in contexto
    assert "Contenido del segundo fragmento." in contexto


def test_formatear_contexto_vacio_devuelve_mensaje_explicito():
    contexto = _formatear_contexto([])
    assert "no se recuper" in contexto.lower()


def test_formatear_contexto_usa_source_si_no_hay_fuente_id():
    documentos = [Document(page_content="Texto.", metadata={"source": "doc_c.md"})]
    contexto = _formatear_contexto(documentos)
    assert "[FUENTE: doc_c.md]" in contexto
