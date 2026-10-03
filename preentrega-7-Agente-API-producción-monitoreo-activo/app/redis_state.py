"""
redis_state.py
===============

Persistencia del ESTADO DE LOS JOBS (no confundir con el checkpointer de
LangGraph de `redis_checkpointer.py`, que persiste el estado INTERNO del
grafo): acá se guarda, por `job_id`, en qué estado está la tarea que pidió
el usuario (`PENDING` -> `RUNNING` -> `DONE`/`FAILED`, con
`AWAITING_APPROVAL` en el medio si el HITL se activó), para que
`GET /tasks/{id}` pueda responder sin bloquear y sin tener que tocar el
grafo para nada.

Cada job es un HASH de Redis `job:{job_id}` con:
- status: uno de `EstadoJob`
- query: la consulta original del usuario
- thread_id: el `thread_id` de LangGraph asociado (mismo valor que
  `job_id` en esta app, pero se guarda aparte para dejarlo explícito)
- created_at / updated_at: timestamps ISO-8601 UTC
- result: (si DONE) la respuesta final del orquestador, en JSON
- error: (si FAILED) el mensaje de error
- approval_payload: (si AWAITING_APPROVAL) el payload del `interrupt(...)`
  de `app/hitl.py`, en JSON — lo que el cliente necesita mostrarle a un
  humano para decidir si aprueba o no
"""

from __future__ import annotations

import json
import time
import uuid
from enum import StrEnum
from typing import Any

from redis.asyncio import Redis

from app.config import JOB_TTL_SEGUNDOS


class EstadoJob(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    DONE = "DONE"
    FAILED = "FAILED"


def _clave(job_id: str) -> str:
    return f"job:{job_id}"


def _ahora() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


async def crear_job(redis: Redis, query: str) -> str:
    """Crea un job nuevo en Redis con status PENDING y devuelve su
    `job_id`. Esto es lo único que hace `POST /tasks` de forma síncrona
    (una escritura a Redis, no al LLM) — por eso responde al instante."""
    job_id = str(uuid.uuid4())
    ahora = _ahora()
    mapping = {
        "status": EstadoJob.PENDING.value,
        "query": query,
        "thread_id": job_id,
        "created_at": ahora,
        "updated_at": ahora,
    }
    await redis.hset(_clave(job_id), mapping=mapping)
    return job_id


async def obtener_job(redis: Redis, job_id: str) -> dict[str, Any] | None:
    crudo = await redis.hgetall(_clave(job_id))
    if not crudo:
        return None
    job: dict[str, Any] = {
        (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
        for k, v in crudo.items()
    }
    for campo in ("result", "approval_payload"):
        if campo in job and job[campo]:
            try:
                job[campo] = json.loads(job[campo])
            except (json.JSONDecodeError, TypeError):
                pass
    return job


async def actualizar_job(redis: Redis, job_id: str, **campos: Any) -> None:
    """Actualiza (merge, no reemplaza) los campos de un job. Los valores
    no-string (dict/list) se serializan a JSON automáticamente."""
    mapping: dict[str, str] = {"updated_at": _ahora()}
    for clave, valor in campos.items():
        if isinstance(valor, (dict, list)):
            mapping[clave] = json.dumps(valor, ensure_ascii=False)
        elif valor is None:
            mapping[clave] = ""
        else:
            mapping[clave] = str(valor)
    await redis.hset(_clave(job_id), mapping=mapping)

    nuevo_status = campos.get("status")
    if JOB_TTL_SEGUNDOS and nuevo_status in (EstadoJob.DONE.value, EstadoJob.FAILED.value, EstadoJob.DONE, EstadoJob.FAILED):
        await redis.expire(_clave(job_id), JOB_TTL_SEGUNDOS)


async def marcar_failed(redis: Redis, job_id: str, error: Exception | str) -> None:
    """Helper específico para el caso que pide explícitamente el
    enunciado: si el background task explota, el job TIENE que quedar
    en FAILED (nunca colgado en RUNNING para siempre, porque si no el
    cliente hace polling eterno a `GET /tasks/{id}`)."""
    await actualizar_job(redis, job_id, status=EstadoJob.FAILED.value, error=str(error))
