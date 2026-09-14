"""
demo_llm.py
===========

LLM determinístico de prueba: en vez de razonar de verdad, devuelve — en
orden — una lista de mensajes fijada de antemano ("guion"). Se usa
ÚNICAMENTE en `demo_offline.py` para generar `trace_ejemplo.json` sin gastar
llamadas a ninguna API (ni siquiera a la gratuita de Groq), de forma
reproducible.

Importante: esto NO reemplaza la prueba real. El grafo, el `ToolNode`, la
condición `tools_condition` y el `AsyncSqliteSaver` que se ejercitan al
correr este LLM falso son exactamente los mismos que se usan con un LLM
real — lo único "de utilería" es qué herramienta se decide llamar en cada
paso. Para generar una traza con razonamiento real (Groq/OpenAI/Anthropic),
corré `run_demo.py` con tu propia API key.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage


class LLMDeterministaDeDemo:
    """Actúa como un `BaseChatModel` (implementa `bind_tools` y `ainvoke`)
    pero en vez de llamar a un modelo real, va entregando los mensajes de
    `guion` en orden, uno por cada paso del nodo "agente"."""

    def __init__(self, guion: list[AIMessage]) -> None:
        self._guion = list(guion)
        self._indice = 0

    def bind_tools(self, herramientas: Any) -> "LLMDeterministaDeDemo":
        return self

    async def ainvoke(self, mensajes: list[BaseMessage]) -> AIMessage:
        if self._indice >= len(self._guion):
            raise RuntimeError(
                "El guion del LLM de demo se quedó sin pasos programados "
                "(¿el grafo entró en un bucle o el escenario necesita un "
                "paso más?)."
            )
        siguiente = self._guion[self._indice]
        self._indice += 1
        return siguiente
