"""
worker.py
=========

Corre el orquestador multi-agente EN SEGUNDO PLANO (nunca dentro del
request/response de un endpoint) y va actualizando el estado del job en
Redis (`PENDING` -> `RUNNING` -> `DONE`/`FAILED`, con `AWAITING_APPROVAL`
en el medio si el HITL se activa) — así `POST /tasks` puede devolver el
`job_id` al instante sin esperar a que el LLM termine.

Dos entradas posibles a este módulo:

- `ejecutar_job(...)`: arranca una tarea NUEVA desde cero (llamado por
  `POST /tasks`).
- `reanudar_job(...)`: retoma una tarea que se había quedado esperando
  aprobación humana (llamado por `POST /tasks/{id}/approve`), usando el
  `Command(resume=...)` de LangGraph — el `RedisSaver` reconstruye el
  estado exacto donde se había frenado.

Manejo de errores (el "error común a evitar" que pide el enunciado):
TODO el cuerpo de ambas funciones corre dentro de un `try/except`
genérico. Si el grafo explota por lo que sea (rate limit de Groq, un bug,
lo que sea), el `except` deja el job en `FAILED` con el mensaje de error
en Redis — nunca se queda trabado en `RUNNING` para siempre, que haría
que un cliente quedase haciendo polling eterno a `GET /tasks/{id}`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langgraph.types import Command
from redis.asyncio import Redis

from app.config import RECURSION_LIMIT
from app.graph import construir_grafo
from app.redis_checkpointer import RedisSaver
from app.redis_state import EstadoJob, actualizar_job, marcar_failed
from app.state import estado_inicial

logger = logging.getLogger(__name__)

# Referencias fuertes a las tareas en vuelo: si no se guardan en algún
# lado, `asyncio` puede recolectar como basura una Task todavía corriendo
# apenas la función que la creó termina (es un gotcha conocido de
# `asyncio.create_task` sin asignar la referencia a una variable que
# sobreviva) y la tarea se cancela a mitad de camino sin ningún aviso.
_tareas_en_vuelo: set[asyncio.Task] = set()


def lanzar_en_segundo_plano(coro) -> asyncio.Task:
    tarea = asyncio.create_task(coro)
    _tareas_en_vuelo.add(tarea)
    tarea.add_done_callback(_tareas_en_vuelo.discard)
    return tarea


def _config_grafo(job_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": job_id}, "recursion_limit": RECURSION_LIMIT}


async def _procesar_resultado(redis: Redis, job_id: str, resultado: dict[str, Any]) -> None:
    if "__interrupt__" in resultado and resultado["__interrupt__"]:
        payload = resultado["__interrupt__"][0].value
        await actualizar_job(
            redis,
            job_id,
            status=EstadoJob.AWAITING_APPROVAL.value,
            approval_payload=payload,
        )
        logger.info("job %s esperando aprobación humana: %s", job_id, payload)
        return

    respuesta_final = resultado["messages"][-1].content
    resultado_job = {
        "respuesta": respuesta_final,
        "pasos_supervisor": resultado.get("pasos_dados"),
        "tarea_completada": resultado.get("tarea_completada"),
        "contribuciones": resultado.get("contribuciones", []),
    }
    await actualizar_job(redis, job_id, status=EstadoJob.DONE.value, result=resultado_job)
    logger.info("job %s completado", job_id)


async def ejecutar_job(redis: Redis, job_id: str, query: str) -> None:
    await actualizar_job(redis, job_id, status=EstadoJob.RUNNING.value)
    try:
        grafo = construir_grafo(checkpointer=RedisSaver(redis))
        resultado = await grafo.ainvoke(estado_inicial(query), config=_config_grafo(job_id))
        await _procesar_resultado(redis, job_id, resultado)
    except Exception as exc:  # noqa: BLE001 - defensivo a propósito, ver docstring
        logger.exception("job %s falló", job_id)
        await marcar_failed(redis, job_id, exc)


async def reanudar_job(redis: Redis, job_id: str, aprobado: bool, comentario: str = "") -> None:
    await actualizar_job(redis, job_id, status=EstadoJob.RUNNING.value)
    try:
        grafo = construir_grafo(checkpointer=RedisSaver(redis))
        resultado = await grafo.ainvoke(
            Command(resume={"aprobado": aprobado, "comentario": comentario}),
            config=_config_grafo(job_id),
        )
        await _procesar_resultado(redis, job_id, resultado)
    except Exception as exc:  # noqa: BLE001
        logger.exception("job %s falló al reanudar", job_id)
        await marcar_failed(redis, job_id, exc)
