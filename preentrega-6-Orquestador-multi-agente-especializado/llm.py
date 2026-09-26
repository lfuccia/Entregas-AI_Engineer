"""
llm.py
======

Selector del LLM según `LLM_PROVIDER` (ver config.py). Import perezoso del
SDK de cada proveedor. Devuelve un cliente NUEVO en cada llamada a
`get_llm()` (a diferencia de la pre-entrega anterior, acá no se cachea un
singleton global): el Supervisor y cada especialista necesitan bindear
herramientas/schemas de salida distintos sobre el mismo modelo base, así
que cada uno pide su propia instancia.
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


def get_llm() -> Any:
    """Devuelve una instancia nueva del chat model configurado según
    `LLM_PROVIDER`. Lanza `RuntimeError` con un mensaje claro si falta la
    API key del proveedor elegido, y `ValueError` si `LLM_PROVIDER` no es
    uno de los soportados."""

    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise RuntimeError(
                "Falta GROQ_API_KEY en tu .env. Conseguí una gratis en "
                "https://console.groq.com/keys y completala."
            )
        from langchain_groq import ChatGroq

        return ChatGroq(model=GROQ_MODEL, temperature=LLM_TEMPERATURE)

    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("Falta OPENAI_API_KEY en tu .env.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=OPENAI_MODEL, temperature=LLM_TEMPERATURE)

    if LLM_PROVIDER == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("Falta ANTHROPIC_API_KEY en tu .env.")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=ANTHROPIC_MODEL, temperature=LLM_TEMPERATURE)

    raise ValueError(
        f"LLM_PROVIDER inválido: {LLM_PROVIDER!r}. Usá 'groq', 'openai' o 'anthropic'."
    )
