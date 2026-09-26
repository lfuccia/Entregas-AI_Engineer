"""
main.py
=======

Uso rápido desde la terminal, con el LLM real configurado en tu `.env`:

    python main.py "Investigá las opiniones sobre el lanzamiento del auricular Aurora X2..."

Si no pasás nada por argumento, te lo pide interactivamente.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from config import RECURSION_LIMIT
from graph import construir_grafo
from state import estado_inicial

logging.basicConfig(level="INFO", format="%(asctime)s | %(levelname)-8s | %(message)s")


async def main() -> None:
    pregunta = " ".join(sys.argv[1:]).strip()
    if not pregunta:
        pregunta = input("Consulta para el orquestador: ").strip()
    if not pregunta:
        print("No diste ninguna consulta.")
        return

    grafo = construir_grafo().compile()
    resultado = await grafo.ainvoke(
        estado_inicial(pregunta),
        config={"recursion_limit": RECURSION_LIMIT},
    )

    print("\n--- Contribuciones ---")
    for c in resultado["contribuciones"]:
        print(f"[{c['agente']}] {c['contenido']}\n")

    print(f"--- Pasos del Supervisor: {resultado['pasos_dados']} ---")


if __name__ == "__main__":
    asyncio.run(main())
