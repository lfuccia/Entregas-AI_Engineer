"""
tests/test_worker.py

Tests de integración (todavía offline, con LLMs determinísticos) de
`app/worker.py`: la cadena completa `crear_job` -> `ejecutar_job` ->
(si hace falta) `reanudar_job`, contra Redis fake, verificando que:

1. Un job que NO pasa por el analista termina `DONE` en una sola pasada.
2. Un job que SÍ pasa por el analista queda `AWAITING_APPROVAL` con el
   payload del HITL guardado en Redis, y `reanudar_job(...)` lo completa.
3. Si el grafo explota (cualquier excepción), el job queda `FAILED` con
   el mensaje de error -- NUNCA se queda trabado en `RUNNING` (el
   "error común a evitar" que pide explícitamente el enunciado: si esto
   fallara, un cliente real quedaría haciendo polling para siempre).
"""

from __future__ import annotations

from fakeredis import aioredis as fakeredis_asyncio
from langchain_core.messages import AIMessage

import app.worker as worker_module
from app.demo_llm import ChatModelDeterminista, SupervisorDeterminista
from app.graph import construir_grafo
from app.redis_state import EstadoJob, crear_job, obtener_job
from app.supervisor import DecisionSupervisor
from app.worker import ejecutar_job, reanudar_job


def _parchear_construir_grafo(monkeypatch, *, supervisor, investigador=None, analista=None):
    def _fake(checkpointer):
        kwargs = {"obtener_llm_supervisor": lambda: supervisor, "checkpointer": checkpointer}
        if investigador is not None:
            kwargs["obtener_llm_investigador"] = lambda: investigador
        if analista is not None:
            kwargs["obtener_llm_analista"] = lambda: analista
        return construir_grafo(**kwargs)

    monkeypatch.setattr(worker_module, "construir_grafo", _fake)


async def test_ejecutar_job_sin_pasar_por_analista_termina_done(monkeypatch):
    redis = fakeredis_asyncio.FakeRedis()
    supervisor_fake = SupervisorDeterminista(
        [DecisionSupervisor(siguiente="FINISH", instruccion="No hacía falta nada más.", razon="Pregunta trivial.")]
    )
    _parchear_construir_grafo(monkeypatch, supervisor=supervisor_fake)

    job_id = await crear_job(redis, "¿Hola?")
    await ejecutar_job(redis, job_id, "¿Hola?")

    job = await obtener_job(redis, job_id)
    assert job["status"] == EstadoJob.DONE.value
    assert job["result"]["respuesta"] == "No hacía falta nada más."
    assert job["result"]["tarea_completada"] is True


async def test_ejecutar_job_que_requiere_analista_queda_awaiting_approval_y_reanudar_lo_completa(monkeypatch):
    redis = fakeredis_asyncio.FakeRedis()
    supervisor_fake = SupervisorDeterminista(
        [
            DecisionSupervisor(siguiente="analista", instruccion="Calculá el promedio.", razon="Falta analizar."),
            DecisionSupervisor(siguiente="FINISH", instruccion="Listo, promedio calculado.", razon="Ya está."),
        ]
    )
    analista_fake = ChatModelDeterminista(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "calcular_metricas", "args": {"numeros": [4, 5]}, "id": "a1"}],
            ),
            AIMessage(content="El promedio es 4.5."),
        ]
    )
    _parchear_construir_grafo(monkeypatch, supervisor=supervisor_fake, analista=analista_fake)

    job_id = await crear_job(redis, "Calculame el promedio de 4 y 5.")
    await ejecutar_job(redis, job_id, "Calculame el promedio de 4 y 5.")

    job = await obtener_job(redis, job_id)
    assert job["status"] == EstadoJob.AWAITING_APPROVAL.value
    assert job["approval_payload"]["tipo"] == "aprobacion_requerida"
    assert job["approval_payload"]["instruccion_pendiente"] == "Calculá el promedio."

    await reanudar_job(redis, job_id, aprobado=True, comentario="Adelante.")

    job = await obtener_job(redis, job_id)
    assert job["status"] == EstadoJob.DONE.value
    assert job["result"]["respuesta"] == "Listo, promedio calculado."


async def test_ejecutar_job_marca_failed_si_el_grafo_explota(monkeypatch):
    redis = fakeredis_asyncio.FakeRedis()

    class _GrafoQueExplota:
        async def ainvoke(self, *args, **kwargs):
            raise RuntimeError("429 Too Many Requests (simulado)")

    monkeypatch.setattr(worker_module, "construir_grafo", lambda checkpointer: _GrafoQueExplota())

    job_id = await crear_job(redis, "¿Algo?")
    await ejecutar_job(redis, job_id, "¿Algo?")

    job = await obtener_job(redis, job_id)
    assert job["status"] == EstadoJob.FAILED.value
    assert "429" in job["error"]


async def test_reanudar_job_marca_failed_si_explota_al_retomar(monkeypatch):
    redis = fakeredis_asyncio.FakeRedis()
    job_id = await crear_job(redis, "¿Algo?")

    class _GrafoQueExplotaAlRetomar:
        async def ainvoke(self, *args, **kwargs):
            raise ValueError("boom al retomar")

    monkeypatch.setattr(worker_module, "construir_grafo", lambda checkpointer: _GrafoQueExplotaAlRetomar())

    await reanudar_job(redis, job_id, aprobado=True)

    job = await obtener_job(redis, job_id)
    assert job["status"] == EstadoJob.FAILED.value
    assert "boom al retomar" in job["error"]
