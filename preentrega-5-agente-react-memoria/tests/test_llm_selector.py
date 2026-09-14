"""
tests/test_llm_selector.py

Tests offline de `llm.get_llm()`: que falle con un mensaje claro si falta
la API key del proveedor elegido, y que rechace un `LLM_PROVIDER` inválido.
No instancia ningún cliente real (no hay red ni API key en estos tests).
"""

from __future__ import annotations

import pytest

import config
import llm


def test_get_llm_sin_groq_api_key_lanza_error_claro(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(llm, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(llm, "GROQ_API_KEY", "")
    llm.resetear_cache_llm()

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        llm.get_llm()


def test_get_llm_proveedor_invalido_lanza_value_error(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "cohere")
    llm.resetear_cache_llm()

    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        llm.get_llm()
