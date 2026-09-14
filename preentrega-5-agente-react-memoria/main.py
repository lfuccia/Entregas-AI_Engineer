"""
main.py
=======

REPL interactivo para probar el agente a mano. Pide un `thread_id` al
arrancar: usá el mismo `thread_id` entre corridas (o entre preguntas de la
misma sesión) para que el agente recuerde la conversación anterior gracias
al checkpointer SQLite; usá uno distinto para arrancar de cero.

Uso:
    python main.py
"""

from __future__ import annotations

import asyncio
import logging

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from config import RECURSION_LIMIT, SQLITE_DB_PATH
from graph import construir_grafo

logging.basicConfig(level="INFO", format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("main")


async def chat() -> None:
    print("Agente ReAct con memoria persistente (SQLite). Escribí 'salir' para terminar.")
    thread_id = input("thread_id a usar (Enter = 'default'): ").strip() or "default"

    async with AsyncSqliteSaver.from_conn_string(SQLITE_DB_PATH) as checkpointer:
        grafo = construir_grafo().compile(checkpointer=checkpointer)
        cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}

        while True:
            try:
                pregunta = input("\nVos: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not pregunta:
                continue
            if pregunta.lower() in {"salir", "exit", "quit"}:
                break

            resultado = await grafo.ainvoke(
                {"messages": [HumanMessage(content=pregunta)]}, config=cfg
            )
            print(f"Agente: {resultado['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(chat())
