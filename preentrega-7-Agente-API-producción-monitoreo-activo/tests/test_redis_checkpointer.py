"""
tests/test_redis_checkpointer.py

Prueba `RedisSaver` (app/redis_checkpointer.py) de punta a punta contra un
Redis real EN MEMORIA (`fakeredis`, protocolo-compatible con
`redis.asyncio`) con un mini-grafo propio que usa `interrupt(...)`:

1. Primer `ainvoke` -> el grafo se frena (HITL) y el estado queda
   persistido en Redis (no en un objeto Python en RAM).
2. Se simula "reiniciar el proceso" armando un cliente Redis y un
   `RedisSaver` COMPLETAMENTE NUEVOS (sin compartir ningún objeto Python
   con el primer run) apuntando a la MISMA instancia de Redis, y se
   confirma que el resume funciona igual -- esto es exactamente lo que
   pasa en la API real: `POST /tasks` y `POST /tasks/{id}/approve` son
   dos requests HTTP (y, en el peor caso, dos procesos) distintos.
"""

from __future__ import annotations

from typing import TypedDict

import fakeredis
import pytest
from fakeredis import aioredis as fakeredis_asyncio
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from app.redis_checkpointer import RedisSaver


class _Estado(TypedDict):
    contador: int
    historial: list[str]


async def _nodo_a(estado: _Estado) -> dict:
    return {"contador": estado["contador"] + 1, "historial": estado["historial"] + ["a"]}


async def _nodo_aprobacion(estado: _Estado) -> dict:
    decision = interrupt({"pregunta": "¿seguimos?", "contador_actual": estado["contador"]})
    return {"historial": estado["historial"] + [f"aprobado={decision}"]}


async def _nodo_b(estado: _Estado) -> dict:
    return {"contador": estado["contador"] + 100, "historial": estado["historial"] + ["b"]}


def _armar_grafo(checkpointer):
    builder = StateGraph(_Estado)
    builder.add_node("a", _nodo_a)
    builder.add_node("aprobacion", _nodo_aprobacion)
    builder.add_node("b", _nodo_b)
    builder.set_entry_point("a")
    builder.add_edge("a", "aprobacion")
    builder.add_edge("aprobacion", "b")
    builder.add_edge("b", END)
    return builder.compile(checkpointer=checkpointer)


@pytest.fixture
def servidor_fake():
    """Un único servidor fakeredis en memoria, compartido entre clientes
    `FakeRedis` independientes (así se puede simular "dos procesos
    distintos hablándole al mismo Redis")."""
    return fakeredis.FakeServer()


async def test_checkpoint_sobrevive_a_un_cliente_redis_nuevo(servidor_fake):
    cliente_1 = fakeredis_asyncio.FakeRedis(server=servidor_fake)
    saver_1 = RedisSaver(cliente_1)
    grafo_1 = _armar_grafo(saver_1)

    config = {"configurable": {"thread_id": "job-123"}}
    resultado = await grafo_1.ainvoke({"contador": 0, "historial": []}, config=config)

    assert "__interrupt__" in resultado
    assert resultado["contador"] == 1
    payload = resultado["__interrupt__"][0].value
    assert payload == {"pregunta": "¿seguimos?", "contador_actual": 1}

    # "Reiniciar el proceso": cliente y checkpointer nuevos, mismo servidor.
    cliente_2 = fakeredis_asyncio.FakeRedis(server=servidor_fake)
    saver_2 = RedisSaver(cliente_2)
    grafo_2 = _armar_grafo(saver_2)

    resultado_final = await grafo_2.ainvoke(Command(resume=True), config=config)

    assert "__interrupt__" not in resultado_final
    assert resultado_final["contador"] == 101
    assert resultado_final["historial"] == ["a", "aprobado=True", "b"]


async def test_dos_threads_distintos_no_se_pisan(servidor_fake):
    cliente = fakeredis_asyncio.FakeRedis(server=servidor_fake)
    saver = RedisSaver(cliente)
    grafo = _armar_grafo(saver)

    config_a = {"configurable": {"thread_id": "thread-a"}}
    config_b = {"configurable": {"thread_id": "thread-b"}}

    r_a = await grafo.ainvoke({"contador": 0, "historial": []}, config=config_a)
    r_b = await grafo.ainvoke({"contador": 1000, "historial": []}, config=config_b)
    assert "__interrupt__" in r_a and "__interrupt__" in r_b

    final_a = await grafo.ainvoke(Command(resume=True), config=config_a)
    final_b = await grafo.ainvoke(Command(resume=False), config=config_b)

    assert final_a["contador"] == 101
    assert final_b["contador"] == 1101
    assert final_a["historial"][-2] == "aprobado=True"
    assert final_b["historial"][-2] == "aprobado=False"


async def test_adelete_thread_borra_todas_las_claves_del_thread(servidor_fake):
    cliente = fakeredis_asyncio.FakeRedis(server=servidor_fake)
    saver = RedisSaver(cliente)
    grafo = _armar_grafo(saver)

    config = {"configurable": {"thread_id": "a-borrar"}}
    await grafo.ainvoke({"contador": 0, "historial": []}, config=config)

    claves_antes = await cliente.keys("lg:*a-borrar*")
    assert claves_antes

    await saver.adelete_thread("a-borrar")

    claves_despues = await cliente.keys("lg:*a-borrar*")
    assert claves_despues == []
