"""
run_demo.py
===========

Igual que `demo_offline.py`, pero con el LLM REAL configurado en tu `.env`
(Groq por defecto — gratis, ver README) en vez del determinístico de
prueba. Corré esto para generar una traza con razonamiento genuino del
modelo y así tener evidencia "de verdad" (no solo el guion fijo).

Requiere que tu `.env` tenga una API key válida para el proveedor elegido
(`LLM_PROVIDER`).

Uso:
    python run_demo.py
    # escribe trace_real.json
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from config import RECURSION_LIMIT, SQLITE_DB_PATH
from graph import construir_grafo

logging.basicConfig(level="INFO", format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("run_demo")

SALIDA = Path("trace_real.json")


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


async def _correr(nombre: str, thread_id: str, preguntas: list[str], grafo) -> dict[str, Any]:
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    resultado: dict[str, Any] = {}
    for pregunta in preguntas:
        logger.info("[%s] Usuario: %s", nombre, pregunta)
        resultado = await grafo.ainvoke({"messages": [HumanMessage(content=pregunta)]}, config=cfg)
        logger.info("[%s] Agente: %s", nombre, resultado["messages"][-1].content)
    return {"thread_id": thread_id, "pasos": [_mensaje_a_dict(m) for m in resultado["messages"]]}


async def main() -> None:
    async with AsyncSqliteSaver.from_conn_string(SQLITE_DB_PATH) as saver:
        grafo = construir_grafo().compile(checkpointer=saver)

        escenario_a = await _correr(
            "A - memoria",
            "real-cliente-102",
            [
                "¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?",
                "¿Y cuál fue el último pedido?",
            ],
            grafo,
        )
        escenario_b = await _correr(
            "B - multi-paso",
            "real-nombre-cliente",
            ["¿Cuántos pedidos tuvo Martina Gómez y por cuánto total?"],
            grafo,
        )
        escenario_c = await _correr(
            "C - aclaración",
            "real-ambiguo",
            [
                "¿Cuántos pedidos tuvo el cliente Gómez?",
                "Al 104, Diego Gómez.",
            ],
            grafo,
        )

    trace = {
        "nota": "Traza generada con run_demo.py usando el LLM real configurado en .env (LLM_PROVIDER).",
        "escenario_a_memoria_persistente": escenario_a,
        "escenario_b_razonamiento_multi_paso": escenario_b,
        "escenario_c_ciclo_de_retorno": escenario_c,
    }
    SALIDA.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Traza escrita en '%s'.", SALIDA)


if __name__ == "__main__":
    asyncio.run(main())
