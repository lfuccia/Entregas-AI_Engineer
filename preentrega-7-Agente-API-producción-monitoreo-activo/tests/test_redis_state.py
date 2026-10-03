"""
tests/test_redis_state.py

Tests del store de estado de jobs (`app/redis_state.py`) contra
`fakeredis`: creación, lectura, actualización (incluyendo la
serialización automática a JSON de `result`/`approval_payload`) y el
helper `marcar_failed` que es, justamente, el mecanismo que evita que un
job se quede trabado en `RUNNING` para siempre si el worker explota.
"""

from __future__ import annotations

from fakeredis import aioredis as fakeredis_asyncio

from app.redis_state import EstadoJob, actualizar_job, crear_job, marcar_failed, obtener_job


async def test_crear_y_obtener_job():
    redis = fakeredis_asyncio.FakeRedis()
    job_id = await crear_job(redis, "¿Qué opinan del producto X?")

    job = await obtener_job(redis, job_id)
    assert job is not None
    assert job["status"] == EstadoJob.PENDING.value
    assert job["query"] == "¿Qué opinan del producto X?"
    assert job["thread_id"] == job_id
    assert "created_at" in job and "updated_at" in job


async def test_obtener_job_inexistente_devuelve_none():
    redis = fakeredis_asyncio.FakeRedis()
    assert await obtener_job(redis, "no-existe") is None


async def test_actualizar_job_serializa_dict_y_list_a_json():
    redis = fakeredis_asyncio.FakeRedis()
    job_id = await crear_job(redis, "consulta")

    await actualizar_job(
        redis,
        job_id,
        status=EstadoJob.DONE.value,
        result={"respuesta": "listo", "contribuciones": [{"agente": "analista", "contenido": "ok"}]},
    )

    job = await obtener_job(redis, job_id)
    assert job["status"] == EstadoJob.DONE.value
    assert isinstance(job["result"], dict)
    assert job["result"]["respuesta"] == "listo"
    assert job["result"]["contribuciones"][0]["agente"] == "analista"


async def test_marcar_failed_deja_el_job_en_failed_con_el_error():
    redis = fakeredis_asyncio.FakeRedis()
    job_id = await crear_job(redis, "consulta")
    await actualizar_job(redis, job_id, status=EstadoJob.RUNNING.value)

    await marcar_failed(redis, job_id, RuntimeError("Groq devolvió 429 demasiadas veces"))

    job = await obtener_job(redis, job_id)
    assert job["status"] == EstadoJob.FAILED.value
    assert "429" in job["error"]


async def test_job_awaiting_approval_guarda_el_payload_del_interrupt():
    redis = fakeredis_asyncio.FakeRedis()
    job_id = await crear_job(redis, "consulta")

    payload = {
        "tipo": "aprobacion_requerida",
        "accion": "analista.calcular_metricas_y_sentimiento",
        "instruccion_pendiente": "Analizá X.",
    }
    await actualizar_job(redis, job_id, status=EstadoJob.AWAITING_APPROVAL.value, approval_payload=payload)

    job = await obtener_job(redis, job_id)
    assert job["status"] == EstadoJob.AWAITING_APPROVAL.value
    assert job["approval_payload"] == payload
