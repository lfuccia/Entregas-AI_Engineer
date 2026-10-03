"""
scripts/demo_offline.py
========================

Corre el flujo COMPLETO de la API (crear job -> ejecutar en segundo
plano -> HITL -> aprobar -> terminar) contra un Redis REAL (no fake),
pero con LLMs determinísticos de `app/demo_llm.py` en vez de Groq/OpenAI/
Anthropic — así podés verificar que TODO el stack (Redis, el
checkpointer, el worker, el nodo de aprobación humana) funciona de
verdad en tu máquina sin gastar ninguna llamada a una API paga ni
necesitar ninguna key.

Requisito: tener un Redis corriendo y accesible en `REDIS_URL` (por
default, `redis://localhost:6379/0` — levantalo con
`docker compose up -d redis` o con un `redis-server` local).

Uso:
    python scripts/demo_offline.py

Al final deja un `trace_demo_offline.json` en la raíz del proyecto con
el detalle completo de la corrida (para inspeccionar sin tener que leer
logs).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redis.asyncio import Redis

from app.config import REDIS_URL
import app.worker as worker_module
from app.demo_scenario import construir_grafo_demo
from app.redis_state import crear_job, obtener_job
from app.worker import ejecutar_job, reanudar_job


async def main() -> None:
    # Se parchea `construir_grafo` DENTRO del módulo worker (igual que en
    # los tests): así `ejecutar_job`/`reanudar_job` -- el código de
    # producción real, sin ningún atajo -- usan los LLMs de guion en vez
    # de llamar a `get_llm()` (que exigiría una API key real).
    worker_module.construir_grafo = construir_grafo_demo

    redis = Redis.from_url(REDIS_URL)
    try:
        await redis.ping()
    except Exception as exc:
        print(f"No se pudo conectar a Redis en {REDIS_URL}: {exc}")
        print("Levantalo con 'docker compose up -d redis' o un 'redis-server' local y reintentá.")
        return

    traza: dict = {"redis_url": REDIS_URL}

    query = "Investigá las opiniones sobre el auricular Aurora X2 y dame el sentimiento general y el puntaje promedio."
    job_id = await crear_job(redis, query)
    traza["job_id"] = job_id
    print(f"1. Job creado en Redis: {job_id} (status=PENDING)\n")

    print("2. Ejecutando el orquestador (investigador -> aprobación humana) ...")
    await ejecutar_job(redis, job_id, query)
    job = await obtener_job(redis, job_id)
    traza["tras_ejecutar_job"] = job
    print(f"   status actual: {job['status']}")
    if job["status"] == "AWAITING_APPROVAL":
        print("   -> el grafo se frenó esperando aprobación humana antes de ejecutar al analista:")
        print(f"      {json.dumps(job['approval_payload'], ensure_ascii=False, indent=6)}\n")

        print("3. Simulando la aprobación humana (POST /tasks/{id}/approve con aprobado=True) ...")
        await reanudar_job(redis, job_id, aprobado=True, comentario="Aprobado por el demo offline.")
        job = await obtener_job(redis, job_id)
        traza["tras_aprobar"] = job
        print(f"   status final: {job['status']}\n")

    print("--- Resultado final ---")
    print(json.dumps(job.get("result", {}), ensure_ascii=False, indent=2))

    salida = Path(__file__).resolve().parent.parent / "trace_demo_offline.json"
    salida.write_text(json.dumps(traza, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTraza completa guardada en {salida}")

    await redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())
