"""
tests/test_llm_selector.py

Tests offline de `llm.get_llm()`: falla con un mensaje claro si falta la
API key del proveedor elegido, y rechaza un `LLM_PROVIDER` inválido. No
instancia ningún cliente real.
"""

from __future__ import annotations

import pytest

import app.llm as llm


def test_get_llm_sin_groq_api_key_lanza_error_claro(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(llm, "GROQ_API_KEY", "")

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        llm.get_llm()


def test_get_llm_proveedor_invalido_lanza_value_error(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "cohere")

    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        llm.get_llm()
