"""
agents/analyst_agent.py
=========================

Agente de Análisis/Cómputo: sub-agente ReAct con dos herramientas acotadas
para procesar datos ya obtenidos (no busca nada por su cuenta):
`analizar_sentimiento` y `calcular_metricas`.
"""

from __future__ import annotations

from typing import Any, Callable

from app.agents._compat import crear_agente_react
from app.llm import get_llm
from app.tools.analisis import analizar_sentimiento, calcular_metricas

PROMPT_ANALISTA = (
    "Sos el Agente de Análisis/Cómputo de un equipo de análisis. Recibís "
    "datos ya obtenidos por el Agente de Investigación (texto de reseñas, "
    "calificaciones u otros números) y tu trabajo es procesarlos con tus "
    "herramientas: `analizar_sentimiento` para opiniones en texto libre, "
    "`calcular_metricas` para cifras (promedios, totales, mínimos, "
    "máximos). NUNCA calcules un promedio ni clasifiques un sentimiento "
    "'a ojo': siempre usá la herramienta correspondiente, así el resultado "
    "es verificable. Si la instrucción pide varias cosas (ej. sentimiento "
    "Y promedio de calificaciones), usá todas las herramientas que hagan "
    "falta antes de responder. Si te piden refinar un análisis anterior "
    "porque quedó incompleto, fijate específicamente qué faltó y agregalo."
)


def construir_agente_analista(obtener_llm: Callable[[], Any] = get_llm):
    """Devuelve el sub-grafo ReAct del analista ya compilado. `obtener_llm`
    es inyectable para poder testear con un LLM falso."""
    return crear_agente_react(
        obtener_llm(),
        tools=[analizar_sentimiento, calcular_metricas],
        prompt=PROMPT_ANALISTA,
        name="analista",
    )
