"""
agents/research_agent.py
=========================

Agente de Investigación: arma un sub-agente ReAct (`create_react_agent` de
LangGraph) con una única herramienta acotada (`buscar_informacion`). No
sabe nada del Supervisor ni del resto del estado compartido — recibe una
instrucción puntual y devuelve su hallazgo; el nodo que lo envuelve en
`graph.py` es el que traduce eso al `EstadoOrquestador` global.
"""

from __future__ import annotations

from typing import Any, Callable

from agents._compat import crear_agente_react
from llm import get_llm
from tools.busqueda import buscar_informacion

PROMPT_INVESTIGADOR = (
    "Sos el Agente de Investigación de un equipo de análisis. Tu único "
    "trabajo es buscar información externa relevante con la herramienta "
    "`buscar_informacion` y reportar lo que encontraste de forma clara y "
    "completa (citando texto y calificación de cada resultado si los hay). "
    "No opines, no calcules promedios ni analices sentimiento — eso lo "
    "hace el Agente de Análisis después. Si la búsqueda no trae nada útil, "
    "decilo explícitamente en vez de inventar datos."
)


def construir_agente_investigador(obtener_llm: Callable[[], Any] = get_llm):
    """Devuelve el sub-grafo ReAct del investigador ya compilado. `obtener_llm`
    es inyectable para poder testear con un LLM falso (ver tests/ y
    demo_llm.py)."""
    return crear_agente_react(
        obtener_llm(),
        tools=[buscar_informacion],
        prompt=PROMPT_INVESTIGADOR,
        name="investigador",
    )
