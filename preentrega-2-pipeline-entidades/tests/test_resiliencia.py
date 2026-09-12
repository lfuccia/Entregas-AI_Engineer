"""
tests/test_resiliencia.py

Tests unitarios (offline, sin llamar a ningún LLM real) para la lógica de
resiliencia de `chain.py`:

    - Detección de `finish_reason` truncado (respuesta cortada por tokens).
    - Detección de JSON mal formado / que no pasó la validación Pydantic.
    - Camino feliz: una respuesta completa y válida se devuelve tal cual.

También verifica que el `ChatPromptTemplate` sea realmente modular (usa
variables de plantilla en vez de f-strings hardcodeadas).
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from chain import (
    PROMPT,
    RespuestaIncompletaError,
    RespuestaMalFormadaError,
    _validar_respuesta_cruda,
)
from schemas import EntidadesTecnicas


def _mensaje_crudo(finish_reason: str | None) -> AIMessage:
    metadata = {}
    if finish_reason is not None:
        metadata["finish_reason"] = finish_reason
    return AIMessage(content="", response_metadata=metadata)


def test_finish_reason_length_dispara_reintento():
    resultado_crudo = {
        "raw": _mensaje_crudo("length"),
        "parsed": None,
        "parsing_error": None,
    }
    with pytest.raises(RespuestaIncompletaError):
        _validar_respuesta_cruda(resultado_crudo)


def test_finish_reason_max_tokens_dispara_reintento():
    resultado_crudo = {
        "raw": _mensaje_crudo("max_tokens"),
        "parsed": None,
        "parsing_error": None,
    }
    with pytest.raises(RespuestaIncompletaError):
        _validar_respuesta_cruda(resultado_crudo)


def test_parsing_error_dispara_reintento():
    resultado_crudo = {
        "raw": _mensaje_crudo("stop"),
        "parsed": None,
        "parsing_error": ValueError("JSON incompleto: falta cerrar llave"),
    }
    with pytest.raises(RespuestaMalFormadaError):
        _validar_respuesta_cruda(resultado_crudo)


def test_parsed_none_sin_error_explicito_tambien_dispara_reintento():
    # Caso borde: no vino parsing_error pero tampoco se pudo parsear.
    resultado_crudo = {
        "raw": _mensaje_crudo("stop"),
        "parsed": None,
        "parsing_error": None,
    }
    with pytest.raises(RespuestaMalFormadaError):
        _validar_respuesta_cruda(resultado_crudo)


def test_respuesta_completa_y_valida_se_devuelve_sin_reintentar():
    objeto_valido = EntidadesTecnicas(
        tecnologias=["FastAPI", "Redis", "PostgreSQL"],
        nivel_de_criticidad="alta",
        resumen_tecnico="API con caché en Redis y persistencia en PostgreSQL.",
    )
    resultado_crudo = {
        "raw": _mensaje_crudo("stop"),
        "parsed": objeto_valido,
        "parsing_error": None,
    }
    resultado = _validar_respuesta_cruda(resultado_crudo)
    assert resultado is objeto_valido


def test_prompt_es_modular_y_no_hardcodea_texto_con_fstrings():
    """El prompt debe declarar sus variables de entrada vía
    ChatPromptTemplate (input_variables), no interpolarlas a mano con
    f-strings de Python antes de construir el template."""
    assert "texto_entrada" in PROMPT.input_variables
    assert "instrucciones_formato" in PROMPT.input_variables
