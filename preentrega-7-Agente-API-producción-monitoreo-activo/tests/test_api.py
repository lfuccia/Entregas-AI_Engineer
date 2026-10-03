"""
tests/test_api.py

Tests de la API de FastAPI de punta a punta (vía ASGI, sin levantar un
socket real) contra Redis fake y LLMs determinísticos: se monkeypatchea
`app.worker.construir_grafo` (la ÚNICA función que construye el grafo
real con `get_llm()`) para que use los LLMs de prueba en vez de llamar a
Groq/OpenAI/Anthropic de verdad.

`httpx.ASGITransport` NO dispara el `lifespan` de la app (no conecta a un
Redis real), así que el fixture `cliente` pisa directamente
`app.state.redis` con un cliente `fakeredis` antes de cada test -- eso
es justo lo que hace falta para testear sin un Redis real corriendo.

Lo que se verifica, con cronometraje real (no solo "no explota"):

1. `POST /tasks` responde con `job_id` AL INSTANTE, sin esperar a que el
   LLM (acá, deliberadamente lento) termine -- la prueba concreta de que
   el event loop no se bloquea.
2. `GET /tasks/{id}` nunca bloquea: devuelve el estado que haya en Redis
   en ese momento (`PENDING`/`RUNNING`), no espera a que termine.
3. El ciclo completo de Human-in-the-loop vía HTTP: `AWAITING_APPROVAL`
   con el payload del interrupt, `POST /tasks/{id}/approve`, y que
   después sí llega a `DONE`.
4. `POST /tasks/{id}/approve` sobre un job que NO está esperando
   aprobación devuelve 409 (no lo deja aprobar dos veces ni aprobar algo
   que ya terminó).
5. `GET /tasks/{id}` sobre un id inexistente devuelve 404.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest
from fakeredis import aioredis as fakeredis_asyncio
from langchain_core.messages import AIMessage

import app.worker as worker_module
from app.demo_llm import ChatModelDeterminista
from app.graph import construir_grafo
from app.main import app
from app.redis_state import EstadoJob, actualizar_job, crear_job
from app.supervisor import DecisionSupervisor


class _SupervisorLento:
    """Igual que `SupervisorDeterminista` pero con una demora real
    (`asyncio.sleep`) antes de cada decisión -- así se puede comprobar
    con cronómetro que la API no se queda esperándolo."""

    def __init__(self, guion: list[DecisionSupervisor], demora: float = 0.3) -> None:
        self._guion = list(guion)
        self._indice = 0
        self._demora = demora

    def with_structured_output(self, esquema):
        return self

    async def ainvoke(self, mensajes):
        await asyncio.sleep(self._demora)
        decision = self._guion[self._indice]
        self._indice += 1
        return decision


def _parchear_construir_grafo(monkeypatch, *, supervisor, analista=None):
    def _fake(checkpointer):
        kwargs = {"obtener_llm_supervisor": lambda: supervisor, "checkpointer": checkpointer}
        if analista is not None:
            kwargs["obtener_llm_analista"] = lambda: analista
        return construir_grafo(**kwargs)

    monkeypatch.setattr(worker_module, "construir_grafo", _fake)


@pytest.fixture
async def cliente():
    redis = fakeredis_asyncio.FakeRedis()
    app.state.redis = redis
    transporte = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transporte, base_url="http://test") as client:
        yield client, redis


async def _esperar_status(client, job_id: str, status_esperado: str, intentos: int = 40, espera: float = 0.05):
    for _ in range(intentos):
        resp = await client.get(f"/tasks/{job_id}")
        if resp.json()["status"] == status_esperado:
            return resp
        await asyncio.sleep(espera)
    raise AssertionError(f"el job {job_id} nunca llegó a status={status_esperado}")


async def test_health(cliente):
    client, _redis = cliente
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["redis"] is True


async def test_post_tasks_responde_al_instante_sin_esperar_al_llm(monkeypatch, cliente):
    client, _redis = cliente
    supervisor_lento = _SupervisorLento(
        [DecisionSupervisor(siguiente="FINISH", instruccion="Listo, sin novedad.", razon="Pregunta trivial.")],
        demora=0.3,
    )
    _parchear_construir_grafo(monkeypatch, supervisor=supervisor_lento)

    t0 = time.monotonic()
    resp = await client.post("/tasks", json={"query": "¿hola?"})
    elapsed_post = time.monotonic() - t0

    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    assert resp.json()["status"] == EstadoJob.PENDING.value
    # El LLM (falso) tarda 0.3s en responder; el POST tiene que volver
    # MUCHO antes que eso -- si este assert fallara, significaría que el
    # endpoint está bloqueando el event loop esperando al grafo.
    assert elapsed_post < 0.2, f"POST /tasks tardó {elapsed_post:.3f}s: ¿se está bloqueando el event loop?"

    t1 = time.monotonic()
    resp2 = await client.get(f"/tasks/{job_id}")
    elapsed_get = time.monotonic() - t1
    assert elapsed_get < 0.1
    assert resp2.json()["status"] in (EstadoJob.PENDING.value, EstadoJob.RUNNING.value)

    resp_final = await _esperar_status(client, job_id, EstadoJob.DONE.value)
    assert resp_final.json()["result"]["respuesta"] == "Listo, sin novedad."


async def test_get_tasks_id_inexistente_da_404(cliente):
    client, _redis = cliente
    resp = await client.get("/tasks/no-existe-este-id")
    assert resp.status_code == 404


async def test_flujo_hitl_completo_via_http(monkeypatch, cliente):
    client, _redis = cliente
    supervisor_fake = _SupervisorLento(
        [
            DecisionSupervisor(siguiente="analista", instruccion="Calculá el promedio de 4 y 5.", razon="Falta analizar."),
            DecisionSupervisor(siguiente="FINISH", instruccion="El promedio es 4.5.", razon="Ya está."),
        ],
        demora=0.01,
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

    resp = await client.post("/tasks", json={"query": "Calculame el promedio de 4 y 5."})
    job_id = resp.json()["job_id"]

    resp_esperando = await _esperar_status(client, job_id, EstadoJob.AWAITING_APPROVAL.value)
    payload = resp_esperando.json()["approval_payload"]
    assert payload["tipo"] == "aprobacion_requerida"
    assert payload["instruccion_pendiente"] == "Calculá el promedio de 4 y 5."

    resp_approve = await client.post(f"/tasks/{job_id}/approve", json={"aprobado": True, "comentario": "dale"})
    assert resp_approve.status_code == 200
    assert resp_approve.json()["status"] == EstadoJob.RUNNING.value

    resp_final = await _esperar_status(client, job_id, EstadoJob.DONE.value)
    assert resp_final.json()["result"]["respuesta"] == "El promedio es 4.5."


async def test_approve_sobre_job_que_no_espera_aprobacion_da_409(cliente):
    client, redis = cliente
    job_id = await crear_job(redis, "consulta")
    await actualizar_job(redis, job_id, status=EstadoJob.DONE.value, result={"respuesta": "ya terminado"})

    resp = await client.post(f"/tasks/{job_id}/approve", json={"aprobado": True})
    assert resp.status_code == 409


async def test_approve_sobre_job_inexistente_da_404(cliente):
    client, _redis = cliente
    resp = await client.post("/tasks/no-existe/approve", json={"aprobado": True})
    assert resp.status_code == 404
