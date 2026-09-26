"""
run_demo.py
===========

Igual que `demo_offline.py`, pero con el LLM REAL configurado en tu `.env`
(Groq por defecto — gratis) para los tres roles (supervisor, investigador,
analista), en vez de los determinísticos de prueba. Corré esto para
generar una traza con razonamiento genuino y así tener evidencia "de
verdad" del flujo de delegación (no solo el guion fijo de
`demo_offline.py`).

Con un LLM real el recorrido puede variar levemente entre corridas (por
ejemplo, el analista podría acertar con sentimiento + promedio en un solo
intento en vez de necesitar el refinamiento) — eso es esperable y está
bien: lo que hay que verificar es que el Supervisor siga habiendo pasado
por los dos especialistas y haya cerrado con `FINISH` antes del
`recursion_limit`.

Uso:
    python run_demo.py "tu consulta acá"
    # si no pasás una consulta, usa una por defecto sobre el Aurora X2.
    # escribe trace_real.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage

from config import RECURSION_LIMIT
from graph import construir_grafo
from state import estado_inicial

logging.basicConfig(level="INFO", format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("run_demo")

SALIDA = Path("trace_real.json")

PREGUNTA_POR_DEFECTO = (
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


async def main() -> None:
    pregunta = " ".join(sys.argv[1:]).strip() or PREGUNTA_POR_DEFECTO
    logger.info("Usuario: %s", pregunta)

    grafo = construir_grafo().compile()
    resultado = await grafo.ainvoke(
        estado_inicial(pregunta),
        config={"recursion_limit": RECURSION_LIMIT},
    )

    for mensaje in resultado["messages"]:
        etiqueta = getattr(mensaje, "name", None) or mensaje.__class__.__name__
        logger.info("[%s] %s", etiqueta, str(mensaje.content)[:300])

    trace = {
        "nota": "Traza generada con run_demo.py usando el LLM real configurado en .env (LLM_PROVIDER).",
        "pregunta": pregunta,
        "pasos_dados_por_el_supervisor": resultado["pasos_dados"],
        "tarea_completada": resultado["tarea_completada"],
        "contribuciones": resultado["contribuciones"],
        "mensajes": [_mensaje_a_dict(m) for m in resultado["messages"]],
    }
    SALIDA.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Traza escrita en '%s'.", SALIDA)


if __name__ == "__main__":
    asyncio.run(main())
