"""
tests/test_schemas.py

Tests unitarios offline (sin LLM, sin embeddings) para `RespuestaRAG`.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas import RespuestaRAG


def test_respuesta_valida_se_construye_correctamente():
    obj = RespuestaRAG(
        respuesta="El rate limiting recomendado es token bucket.",
        fuentes=["03_seguridad_y_autenticacion.md#fragmento_1"],
    )
    assert "token bucket" in obj.respuesta
    assert obj.fuentes == ["03_seguridad_y_autenticacion.md#fragmento_1"]


def test_fuentes_por_defecto_es_lista_vacia():
    obj = RespuestaRAG(respuesta="No tengo esa información en los documentos.")
    assert obj.fuentes == []


def test_respuesta_vacia_lanza_excepcion():
    with pytest.raises(ValidationError):
        RespuestaRAG(respuesta="   ", fuentes=[])


def test_fuentes_duplicadas_se_deduplican():
    obj = RespuestaRAG(
        respuesta="Texto de respuesta.",
        fuentes=["a.md#fragmento_0", "a.md#fragmento_0", "b.md#fragmento_2"],
    )
    assert obj.fuentes == ["a.md#fragmento_0", "b.md#fragmento_2"]
