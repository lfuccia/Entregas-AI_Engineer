"""
demo_offline.py
================

Genera `trace_ejemplo.json`: la traza de ejecución ReAct que pide el
checklist de la pre-entrega, cubriendo los tres criterios de aceptación en
tres escenarios separados (cada uno con su propio `thread_id`):

    A) Resiliencia de estado / memoria: dos preguntas en el mismo thread_id;
       la segunda ("¿y el último pedido?") se responde usando datos que ya
       están en el historial, sin volver a llamar a la herramienta.
    B) Razonamiento multi-paso: una sola pregunta que obliga a llamar a DOS
       herramientas distintas encadenadas (buscar_cliente -> buscar_pedidos)
       para poder responder.
    C) Ciclo de retorno: la herramienta devuelve información ambigua/
       incompleta (varios clientes "Gómez") y el agente, en vez de
       inventar, le pide una aclaración al usuario; en el turno siguiente
       (mismo thread_id) retoma con la aclaración.

Corre con un LLM DETERMINÍSTICO de prueba (`demo_llm.py`), no uno real: así
el repo tiene una traza de evidencia reproducible sin depender de ninguna
API key. Para una traza con razonamiento real de un LLM de verdad, usá
`run_demo.py` con tu propia GROQ_API_KEY (u OpenAI/Anthropic).

Uso:
    python demo_offline.py
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from demo_llm import LLMDeterministaDeDemo
from graph import construir_grafo

logging.basicConfig(level="INFO", format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("demo_offline")

RECURSION_LIMIT_DEMO = 10
DB_DEMO = "demo_checkpoints.sqlite"
SALIDA = Path("trace_ejemplo.json")


def _mensaje_a_dict(mensaje: BaseMessage) -> dict[str, Any]:
    resultado: dict[str, Any] = {"tipo": mensaje.__class__.__name__, "contenido": mensaje.content}
    tool_calls = getattr(mensaje, "tool_calls", None)
    if tool_calls:
        resultado["tool_calls"] = [
            {"herramienta": tc["name"], "args": tc["args"]} for tc in tool_calls
        ]
    if isinstance(mensaje, ToolMessage):
        resultado["herramienta"] = mensaje.name
        resultado["tool_call_id"] = mensaje.tool_call_id
    return resultado


async def _correr_escenario(
    nombre: str,
    descripcion: str,
    thread_id: str,
    guion: list[AIMessage],
    preguntas: list[str],
    saver: AsyncSqliteSaver,
) -> dict[str, Any]:
    modelo_falso = LLMDeterministaDeDemo(guion)
    grafo = construir_grafo(obtener_llm=lambda: modelo_falso).compile(checkpointer=saver)
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT_DEMO}

    resultado: dict[str, Any] = {}
    for pregunta in preguntas:
        logger.info("[%s] Usuario: %s", nombre, pregunta)
        resultado = await grafo.ainvoke({"messages": [HumanMessage(content=pregunta)]}, config=cfg)
        respuesta_final = resultado["messages"][-1].content
        logger.info("[%s] Agente: %s", nombre, respuesta_final)

    return {
        "descripcion": descripcion,
        "thread_id": thread_id,
        "pasos": [_mensaje_a_dict(m) for m in resultado["messages"]],
    }


async def generar_trace() -> dict[str, Any]:
    Path(DB_DEMO).unlink(missing_ok=True)

    async with AsyncSqliteSaver.from_conn_string(DB_DEMO) as saver:
        # -- A) Resiliencia de estado: memoria dentro del mismo thread_id --
        escenario_a = await _correr_escenario(
            nombre="A",
            descripcion=(
                "Memoria persistente: la segunda pregunta se responde con "
                "datos ya obtenidos en la primera, sin volver a llamar a la "
                "herramienta, gracias al checkpointer + thread_id."
            ),
            thread_id="demo-cliente-102",
            guion=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "buscar_pedidos", "args": {"cliente_id": 102}, "id": "call_1"}],
                ),
                AIMessage(content="El cliente 102 tuvo 3 pedidos por un total de $14.500."),
                AIMessage(
                    content=(
                        "El último pedido del cliente 102 fue el P-2044, del "
                        "2026-08-30, por $5.200."
                    )
                ),
            ],
            preguntas=[
                "¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?",
                "¿Y cuál fue el último pedido?",
            ],
            saver=saver,
        )

        # -- B) Razonamiento multi-paso: 2 herramientas encadenadas --------
        escenario_b = await _correr_escenario(
            nombre="B",
            descripcion=(
                "Razonamiento multi-paso: para responder una sola pregunta, "
                "el agente llama a buscar_cliente (resolver nombre -> ID) y "
                "después a buscar_pedidos (ID -> pedidos), dos llamadas a "
                "herramientas distintas antes de concluir."
            ),
            thread_id="demo-nombre-cliente",
            guion=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "buscar_cliente", "args": {"nombre_o_id": "Martina Gómez"}, "id": "call_2"}
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "buscar_pedidos", "args": {"cliente_id": 103}, "id": "call_3"}],
                ),
                AIMessage(content="Martina Gómez (cliente 103) tuvo 7 pedidos por un total de $58.900."),
            ],
            preguntas=["¿Cuántos pedidos tuvo Martina Gómez y por cuánto total?"],
            saver=saver,
        )

        # -- C) Ciclo de retorno: aclaración ante info ambigua --------------
        escenario_c = await _correr_escenario(
            nombre="C",
            descripcion=(
                "Ciclo de retorno: buscar_cliente devuelve varios candidatos "
                "ambiguos ('Gómez' matchea a dos clientes) y el agente, en "
                "vez de inventar, le pide al usuario que aclare. En el turno "
                "siguiente (mismo thread_id) retoma con la aclaración."
            ),
            thread_id="demo-ambiguo",
            guion=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "buscar_cliente", "args": {"nombre_o_id": "Gómez"}, "id": "call_4"}],
                ),
                AIMessage(
                    content=(
                        "Hay más de un cliente con apellido Gómez: Martina "
                        "Gómez (103) y Diego Gómez (104). ¿A cuál te referís?"
                    )
                ),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "buscar_pedidos", "args": {"cliente_id": 104}, "id": "call_5"}],
                ),
                AIMessage(content="Diego Gómez (cliente 104) no tiene pedidos registrados todavía."),
            ],
            preguntas=[
                "¿Cuántos pedidos tuvo el cliente Gómez?",
                "Al 104, Diego Gómez.",
            ],
            saver=saver,
        )

    return {
        "nota": (
            "Traza generada con demo_offline.py usando un LLM DETERMINÍSTICO "
            "de prueba (demo_llm.py) — sin llamadas a ninguna API — para que "
            "quede una evidencia reproducible en el repo. El grafo, el "
            "ToolNode, tools_condition y el AsyncSqliteSaver que se ven acá "
            "son los reales; para una traza con razonamiento de un LLM de "
            "verdad, correr run_demo.py con tu propia API key."
        ),
        "escenario_a_memoria_persistente": escenario_a,
        "escenario_b_razonamiento_multi_paso": escenario_b,
        "escenario_c_ciclo_de_retorno": escenario_c,
    }


def main() -> None:
    trace = asyncio.run(generar_trace())
    SALIDA.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Traza escrita en '%s'.", SALIDA)


if __name__ == "__main__":
    main()
