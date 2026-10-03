"""
main.py (FastAPI)
==================

API REST de producción para el orquestador multi-agente de la
pre-entrega 6. Expone:

- `POST /tasks`            -> encola la consulta y devuelve `job_id` AL
                               INSTANTE (nunca espera al LLM).
- `GET  /tasks/{job_id}`   -> estado actual del job, sin bloquear.
- `POST /tasks/{job_id}/approve`
                           -> aprueba o rechaza la acción crítica que el
                              HITL dejó esperando (ver `app/hitl.py`).
- `GET  /health`           -> chequeo rápido de que la API y Redis están
                              vivos (para probes de Docker/orquestadores).

Todo el trabajo pesado (invocar al grafo, que internamente llama al LLM
real varias veces) corre en `app/worker.py`, lanzado como una tarea de
asyncio en segundo plano — el event loop de FastAPI NUNCA se bloquea
esperando una respuesta del LLM ni de Redis (todo acá es async de
punta a punta: `redis.asyncio`, `grafo.ainvoke`, etc.).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from redis.asyncio import Redis

import app.worker as worker_module
from app.config import DEMO_MODE, REDIS_URL
from app.observability import init_observability
from app.redis_state import EstadoJob, actualizar_job, crear_job, obtener_job
from app.worker import ejecutar_job, lanzar_en_segundo_plano, reanudar_job

logging.basicConfig(level="INFO", format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    mensaje_observabilidad = init_observability()
    logger.info(mensaje_observabilidad)

    if DEMO_MODE:
        # DEMO_MODE=true en el .env: usa LLMs determinísticos de guion
        # (ver app/demo_scenario.py) en vez de llamar a Groq/OpenAI/
        # Anthropic de verdad. Sirve para probar la API entera (incluido
        # el HITL) sin ninguna API key. Para la prueba de carga de 5
        # pedidos concurrentes que pide el enunciado hace falta un LLM
        # real -> poner DEMO_MODE=false.
        from app.demo_scenario import construir_grafo_demo

        worker_module.construir_grafo = construir_grafo_demo
        logger.warning(
            "DEMO_MODE=true: usando LLMs de guion (sin API key), NO un LLM real. "
            "Poné DEMO_MODE=false en tu .env antes de la prueba de carga real."
        )

    app.state.redis = Redis.from_url(REDIS_URL, decode_responses=False)
    try:
        await app.state.redis.ping()
        logger.info("conectado a Redis en %s", REDIS_URL)
    except Exception as exc:
        logger.error("no se pudo conectar a Redis (%s): %s", REDIS_URL, exc)

    yield

    await app.state.redis.aclose()


app = FastAPI(
    title="API del Orquestador Multi-Agente",
    description=(
        "API de producción (pre-entrega 7) que expone el sistema "
        "multi-agente Supervisor/Investigador/Analista de la pre-entrega 6, "
        "con jobs asíncronos, persistencia en Redis, observabilidad activa y "
        "un punto de aprobación humana (HITL)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def get_redis(request: Request) -> Redis:
    return request.app.state.redis


# -- esquemas de entrada/salida -------------------------------------------


class NuevaTarea(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="La consulta en lenguaje natural para el orquestador multi-agente.",
        examples=["Investigá las opiniones sobre el auricular Aurora X2 y calculame el promedio de puntaje."],
    )


class RespuestaJobCreado(BaseModel):
    job_id: str
    status: str


class DecisionAprobacion(BaseModel):
    aprobado: bool = Field(..., description="True para aprobar la acción crítica pendiente, False para rechazarla.")
    comentario: str = Field("", description="Comentario opcional del revisor humano (queda registrado en el log del job).")


# -- endpoints --------------------------------------------------------------


@app.get("/health")
async def health(redis: Redis = Depends(get_redis)) -> dict[str, Any]:
    try:
        await redis.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    return {"status": "ok" if redis_ok else "degraded", "redis": redis_ok}


@app.post("/tasks", response_model=RespuestaJobCreado, status_code=202)
async def crear_tarea(tarea: NuevaTarea, redis: Redis = Depends(get_redis)) -> RespuestaJobCreado:
    """Encola la consulta y devuelve el `job_id` de inmediato. La
    ejecución real (que puede tardar varios segundos/minutos con un LLM
    real) corre en segundo plano vía `app/worker.py`."""
    job_id = await crear_job(redis, tarea.query)
    lanzar_en_segundo_plano(ejecutar_job(redis, job_id, tarea.query))
    return RespuestaJobCreado(job_id=job_id, status=EstadoJob.PENDING.value)


@app.get("/tasks/{job_id}")
async def obtener_tarea(job_id: str, redis: Redis = Depends(get_redis)) -> dict[str, Any]:
    """Devuelve el estado actual del job sin bloquear: es una simple
    lectura a Redis, nunca espera a que el grafo termine."""
    job = await obtener_job(redis, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No existe ningún job con id '{job_id}'.")
    return job


@app.post("/tasks/{job_id}/approve")
async def aprobar_tarea(
    job_id: str, decision: DecisionAprobacion, redis: Redis = Depends(get_redis)
) -> dict[str, Any]:
    """Resuelve el punto de Human-in-the-loop de un job que quedó en
    `AWAITING_APPROVAL`: reanuda el grafo exactamente donde se había
    frenado (vía el checkpoint guardado en Redis), aprobando o
    rechazando la acción crítica pendiente (ver `app/hitl.py`)."""
    job = await obtener_job(redis, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No existe ningún job con id '{job_id}'.")
    if job["status"] != EstadoJob.AWAITING_APPROVAL.value:
        raise HTTPException(
            status_code=409,
            detail=(
                f"El job '{job_id}' no está esperando aprobación (status actual: "
                f"'{job['status']}')."
            ),
        )

    await actualizar_job(redis, job_id, status=EstadoJob.RUNNING.value)
    lanzar_en_segundo_plano(reanudar_job(redis, job_id, decision.aprobado, decision.comentario))
    return {"job_id": job_id, "status": EstadoJob.RUNNING.value}
