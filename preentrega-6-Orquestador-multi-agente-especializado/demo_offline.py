"""
demo_offline.py
================

Genera `trace_ejemplo.json`: la traza de ejecución del orquestador
completo para UNA consulta que obliga a pasar por los dos especialistas y
por un ciclo de refinamiento antes de cerrar:

    "Investigá las opiniones sobre el lanzamiento del auricular Aurora X2,
    analizá el sentimiento general y el promedio de las calificaciones, y
    dame un resumen."

Recorrido (ver el detalle en cada paso del guion, más abajo):

    supervisor -> investigador -> supervisor -> analista (incompleto:
    solo sentimiento) -> supervisor (pide refinar: falta el promedio) ->
    analista (refinado: agrega el promedio) -> supervisor -> FIN

Corre con LLMs DETERMINÍSTICOS de prueba (`demo_llm.py`) para los tres
roles (supervisor, investigador, analista) — sin llamadas a ninguna API —
así el repo tiene una evidencia reproducible del flujo de delegación
completo. El grafo, el `StateGraph`, el `ToolNode` interno de cada
sub-agente ReAct y las herramientas (`buscar_informacion`,
`analizar_sentimiento`, `calcular_metricas`) que se ven acá son los
reales. Para una corrida con razonamiento real de un LLM de verdad, usá
`run_demo.py` con tu propia API key.

Uso:
    python demo_offline.py
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from demo_llm import ChatModelDeterminista, SupervisorDeterminista
from graph import construir_grafo
from state import estado_inicial
from supervisor import DecisionSupervisor
from tools.busqueda import _RESENAS

logging.basicConfig(level="INFO", format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("demo_offline")

RECURSION_LIMIT_DEMO = 15
SALIDA = Path("trace_ejemplo.json")

PREGUNTA = (
    "Investigá las opiniones sobre el lanzamiento del auricular Aurora X2, "
    "analizá el sentimiento general y el promedio de las calificaciones, y "
    "dame un resumen."
)


def _mensaje_a_dict(mensaje: BaseMessage) -> dict[str, Any]:
    resultado: dict[str, Any] = {
        "tipo": mensaje.__class__.__name__,
        "de": getattr(mensaje, "name", None),
        "contenido": mensaje.content,
    }
    tool_calls = getattr(mensaje, "tool_calls", None)
    if tool_calls:
        resultado["tool_calls"] = [{"herramienta": tc["name"], "args": tc["args"]} for tc in tool_calls]
    if isinstance(mensaje, ToolMessage):
        resultado["herramienta"] = mensaje.name
    return resultado


def _armar_guiones() -> tuple[SupervisorDeterminista, ChatModelDeterminista, ChatModelDeterminista]:
    textos_resenas = [r["texto"] for r in _RESENAS]
    calificaciones = [r["calificacion"] for r in _RESENAS]

    hallazgo_investigador = "Encontré 5 reseñas sobre el lanzamiento del auricular Aurora X2:\n" + "\n".join(
        f"- ({r['calificacion']}/5, {r['fuente']}): {r['texto']}" for r in _RESENAS
    )

    # -- Supervisor: 4 decisiones -------------------------------------------
    guion_supervisor = [
        DecisionSupervisor(
            siguiente="investigador",
            instruccion="Buscá reseñas y opiniones sobre el lanzamiento del auricular Aurora X2.",
            razon="Todavía no hay ninguna contribución: hay que investigar primero.",
        ),
        DecisionSupervisor(
            siguiente="analista",
            instruccion=(
                "Con las reseñas que encontró el investigador, calculá el "
                "sentimiento general Y el promedio de las calificaciones "
                "numéricas."
            ),
            razon="Ya hay datos del investigador; falta el análisis completo (sentimiento + promedio).",
        ),
        DecisionSupervisor(
            siguiente="analista",
            instruccion=(
                "Te faltó calcular el promedio de las calificaciones "
                "numéricas (1 a 5) de las reseñas — agregalo usando "
                "calcular_metricas."
            ),
            razon=(
                "El análisis cubrió el sentimiento pero no la cifra numérica "
                "que pedía el usuario: todavía no es suficiente, hace falta "
                "refinar."
            ),
        ),
        DecisionSupervisor(
            siguiente="FINISH",
            instruccion=(
                "Resumen final: las opiniones sobre el Aurora X2 son, en "
                "general, positivas (sentimiento positivo, score 0.47), con "
                "un promedio de calificación de 3.8/5 sobre 5 reseñas."
            ),
            razon="Ya hay investigación y un análisis completo (sentimiento y promedio) que cubre lo pedido.",
        ),
    ]

    # -- Investigador: 1 llamada a herramienta + respuesta final ------------
    guion_investigador = ChatModelDeterminista(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "buscar_informacion",
                        "args": {"query": "opiniones lanzamiento auricular Aurora X2"},
                        "id": "call_busqueda_1",
                    }
                ],
            ),
            AIMessage(content=hallazgo_investigador),
        ]
    )

    # -- Analista: 2 intentos (incompleto, después refinado) ----------------
    guion_analista = ChatModelDeterminista(
        responses=[
            # Intento 1: solo sentimiento (a propósito, incompleto).
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "analizar_sentimiento",
                        "args": {"textos": textos_resenas},
                        "id": "call_sentimiento_1",
                    }
                ],
            ),
            AIMessage(content="El sentimiento general de las reseñas es positivo (score 0.47)."),
            # Intento 2 (refinado): agrega el promedio pedido.
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calcular_metricas",
                        "args": {"numeros": calificaciones},
                        "id": "call_metricas_1",
                    }
                ],
            ),
            AIMessage(
                content=(
                    "Sentimiento general positivo (score 0.47) y promedio de "
                    "calificaciones 3.8 sobre 5 (n=5)."
                )
            ),
        ]
    )

    return SupervisorDeterminista(guion_supervisor), guion_investigador, guion_analista


async def generar_trace() -> dict[str, Any]:
    supervisor_fake, investigador_fake, analista_fake = _armar_guiones()

    grafo = construir_grafo(
        obtener_llm_supervisor=lambda: supervisor_fake,
        obtener_llm_investigador=lambda: investigador_fake,
        obtener_llm_analista=lambda: analista_fake,
    ).compile()

    logger.info("Usuario: %s", PREGUNTA)
    resultado = await grafo.ainvoke(
        estado_inicial(PREGUNTA),
        config={"recursion_limit": RECURSION_LIMIT_DEMO},
    )

    for mensaje in resultado["messages"]:
        etiqueta = getattr(mensaje, "name", None) or mensaje.__class__.__name__
        logger.info("[%s] %s", etiqueta, str(mensaje.content)[:200])

    return {
        "nota": (
            "Traza generada con demo_offline.py usando LLMs DETERMINÍSTICOS "
            "de prueba (demo_llm.py) para los tres roles (supervisor, "
            "investigador, analista) — sin llamadas a ninguna API — para "
            "que quede una evidencia reproducible en el repo. El grafo, el "
            "StateGraph, el ToolNode interno de cada sub-agente ReAct y las "
            "herramientas que se ven acá son los reales. Para una corrida "
            "con razonamiento real, correr run_demo.py con tu propia API key."
        ),
        "pregunta": PREGUNTA,
        "pasos_dados_por_el_supervisor": resultado["pasos_dados"],
        "tarea_completada": resultado["tarea_completada"],
        "contribuciones": resultado["contribuciones"],
        "mensajes": [_mensaje_a_dict(m) for m in resultado["messages"]],
    }


def main() -> None:
    trace = asyncio.run(generar_trace())
    SALIDA.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Traza escrita en '%s'.", SALIDA)


if __name__ == "__main__":
    main()
