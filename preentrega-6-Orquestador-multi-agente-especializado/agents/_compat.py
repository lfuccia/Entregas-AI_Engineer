"""
agents/_compat.py
==================

`create_react_agent` vive en `langgraph.prebuilt` (que es lo que pide el
enunciado de esta pre-entrega), pero LangGraph 1.x ya lo marca como
deprecated en favor de `langchain.agents.create_agent` (con el kwarg
renombrado `prompt` -> `system_prompt`), y avisa que lo va a eliminar en
V2. Se intenta primero el clásico — el que se pide explícitamente — y,
si una versión futura de LangGraph lo saca del todo, se cae solo al
nuevo, así el proyecto sigue funcionando con cualquiera de las dos series
de versiones sin tocar código (mismo patrón que la relocación de
`EnsembleRetriever` en la pre-entrega 4).
"""

from __future__ import annotations

from typing import Any, Sequence


def crear_agente_react(model: Any, tools: Sequence[Any], prompt: str, name: str) -> Any:
    try:
        from langgraph.prebuilt import create_react_agent

        return create_react_agent(model, tools=tools, prompt=prompt, name=name)
    except ImportError:
        from langchain.agents import create_agent

        return create_agent(model, tools=tools, system_prompt=prompt, name=name)
