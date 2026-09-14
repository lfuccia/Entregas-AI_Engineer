"""
llm.py
======

Selector del LLM según `LLM_PROVIDER` (ver config.py). Import perezoso del
SDK de cada proveedor: así, si estás usando Groq, no hace falta tener
instalado (ni configurada la key de) OpenAI o Anthropic, y viceversa.

Cacheado con un singleton simple (`_llm_cache`) para no reconstruir el
cliente en cada paso del grafo.
"""

from __future__ import annotations

from typing import Any

from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

_llm_cache: Any = None


def get_llm() -> Any:
    """Devuelve el chat model configurado (`BaseChatModel` de LangChain)
    según `LLM_PROVIDER`. Lanza `RuntimeError` con un mensaje claro si falta
    la API key del proveedor elegido, y `ValueError` si `LLM_PROVIDER` no es
    uno de los soportados."""
    global _llm_cache
    if _llm_cache is not None:
        return _llm_cache

    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise RuntimeError(
                "Falta GROQ_API_KEY en tu .env. Conseguí una gratis en "
                "https://console.groq.com/keys y completala."
            )
        from langchain_groq import ChatGroq

        _llm_cache = ChatGroq(model=GROQ_MODEL, temperature=LLM_TEMPERATURE)

    elif LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("Falta OPENAI_API_KEY en tu .env.")
        from langchain_openai import ChatOpenAI

        _llm_cache = ChatOpenAI(model=OPENAI_MODEL, temperature=LLM_TEMPERATURE)

    elif LLM_PROVIDER == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("Falta ANTHROPIC_API_KEY en tu .env.")
        from langchain_anthropic import ChatAnthropic

        _llm_cache = ChatAnthropic(model=ANTHROPIC_MODEL, temperature=LLM_TEMPERATURE)

    else:
        raise ValueError(
            f"LLM_PROVIDER inválido: {LLM_PROVIDER!r}. Usá 'groq', 'openai' o "
            "'anthropic'."
        )

    return _llm_cache


def resetear_cache_llm() -> None:
    """Solo para tests: fuerza a que la próxima `get_llm()` reconstruya el
    cliente en vez de devolver el cacheado."""
    global _llm_cache
    _llm_cache = None
