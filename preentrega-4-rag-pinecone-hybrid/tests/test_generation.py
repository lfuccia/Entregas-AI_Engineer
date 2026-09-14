"""
tests/test_generation.py

Tests unitarios offline (sin GROQ_API_KEY, sin llamar a ningún LLM) para
las funciones puras de `generation.py`: formateo de contexto y extracción
de fuentes. `_get_llm_generacion()` solo se invoca dentro de
`generar_respuesta()`/`agenerar_respuesta()`, nunca al importar el módulo
ni en estas funciones auxiliares.
"""

from __future__ import annotations

from langchain_core.documents import Document

from generation import _formatear_contexto, _fuentes_de


def test_formatear_contexto_incluye_etiquetas_de_fuente():
    documentos = [
        Document(page_content="Contenido A.", metadata={"fuente_id": "a.md#fragmento_0"}),
        Document(page_content="Contenido B.", metadata={"fuente_id": "b.md#fragmento_1"}),
    ]
    contexto = _formatear_contexto(documentos)

    assert "[FUENTE: a.md#fragmento_0]" in contexto
    assert "[FUENTE: b.md#fragmento_1]" in contexto
    assert "Contenido A." in contexto


def test_formatear_contexto_vacio_devuelve_mensaje_explicito():
    contexto = _formatear_contexto([])
    assert "no se recuper" in contexto.lower()


def test_formatear_contexto_usa_source_si_no_hay_fuente_id():
    documentos = [Document(page_content="Texto.", metadata={"source": "c.md"})]
    assert "[FUENTE: c.md]" in _formatear_contexto(documentos)


def test_fuentes_de_deduplica_preservando_orden():
    documentos = [
        Document(page_content="1", metadata={"fuente_id": "a.md#0"}),
        Document(page_content="2", metadata={"fuente_id": "b.md#0"}),
        Document(page_content="3", metadata={"fuente_id": "a.md#0"}),
    ]
    assert _fuentes_de(documentos) == ["a.md#0", "b.md#0"]


def test_generar_respuesta_sin_groq_api_key_lanza_error_claro(monkeypatch):
    import config
    import generation

    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    monkeypatch.setattr(generation, "GROQ_API_KEY", "")

    import pytest

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        generation.generar_respuesta("¿pregunta?", [])
