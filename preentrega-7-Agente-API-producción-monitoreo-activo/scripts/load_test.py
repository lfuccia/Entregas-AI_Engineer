"""
scripts/load_test.py
======================

Dispara N pedidos CONCURRENTES a la API (`POST /tasks`), hace polling de
cada uno hasta que termina (`DONE` o `FAILED`) y calcula la latencia p95
de principio a fin. Es la prueba de carga de 5 pedidos concurrentes que
pide el enunciado.

Uso típico (con la API ya levantada, Redis corriendo, y un LLM real
configurado en el `.env` -- NO en `DEMO_MODE`, para que la latencia y el
costo sean reales):

    python scripts/load_test.py --n 5

Esto imprime un resumen en la terminal y lo guarda en
`screenshots/load_test_resultados.json`. OJO: este script mide la
latencia END-TO-END desde acá afuera (nada que ver con Phoenix/LangSmith)
-- el COSTO por ejecución y el p95 que hay que capturar en pantalla para
`/screenshots` salen del dashboard de observabilidad (ver README), no de
este JSON. Este script sirve para generar la carga real que después
aparece en ese dashboard, y de paso te da un número propio de latencia
para contrastar.

Si algún job termina en HITL (`AWAITING_APPROVAL`, porque el supervisor
decidió mandar al analista), este script lo aprueba automáticamente
(`--auto-approve`, activado por default) para que la prueba de carga no
se quede esperando a un humano -- en un uso real, estás vos mirando el
dashboard y aprobando a mano.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

QUERIES_POR_DEFECTO = [
    "Investigá las opiniones sobre el auricular Aurora X2 y dame el sentimiento general.",
    "¿Qué dicen las reseñas del Aurora X2 sobre la batería? Calculá también el puntaje promedio.",
    "Analizá el sentimiento de las opiniones del Aurora X2 y su puntaje promedio.",
    "Resumime qué opinan del Aurora X2 e indicame el promedio de calificación.",
    "¿Cómo fue recibido el Aurora X2? Quiero sentimiento general y promedio numérico.",
]


async def _correr_un_pedido(client: httpx.AsyncClient, query: str, auto_approve: bool, timeout_s: float) -> dict:
    t0 = time.monotonic()
    resp = await client.post("/tasks", json={"query": query})
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    t_post = time.monotonic() - t0

    aprobado_en = None
    inicio = time.monotonic()
    while True:
        if time.monotonic() - inicio > timeout_s:
            return {
                "job_id": job_id,
                "query": query,
                "status": "TIMEOUT_DEL_SCRIPT",
                "latencia_post_s": t_post,
                "latencia_total_s": time.monotonic() - t0,
            }
        resp = await client.get(f"/tasks/{job_id}")
        job = resp.json()
        status = job["status"]

        if status == "AWAITING_APPROVAL" and auto_approve and aprobado_en is None:
            aprobado_en = time.monotonic() - t0
            await client.post(f"/tasks/{job_id}/approve", json={"aprobado": True, "comentario": "auto-aprobado por load_test.py"})

        if status in ("DONE", "FAILED"):
            return {
                "job_id": job_id,
                "query": query,
                "status": status,
                "latencia_post_s": round(t_post, 4),
                "latencia_total_s": round(time.monotonic() - t0, 4),
                "paso_por_hitl": aprobado_en is not None,
                "error": job.get("error"),
            }
        await asyncio.sleep(0.2)


def _percentil(valores: list[float], p: float) -> float:
    if not valores:
        return 0.0
    valores_ordenados = sorted(valores)
    k = (len(valores_ordenados) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(valores_ordenados) - 1)
    if f == c:
        return valores_ordenados[f]
    return valores_ordenados[f] + (valores_ordenados[c] - valores_ordenados[f]) * (k - f)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL de la API (default: http://localhost:8000)")
    parser.add_argument("--n", type=int, default=5, help="Cantidad de pedidos concurrentes (default: 5)")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout por pedido, en segundos (default: 120)")
    parser.add_argument("--no-auto-approve", action="store_true", help="No aprobar automáticamente los HITL (quedan AWAITING_APPROVAL)")
    args = parser.parse_args()

    queries = (QUERIES_POR_DEFECTO * ((args.n // len(QUERIES_POR_DEFECTO)) + 1))[: args.n]

    print(f"Disparando {args.n} pedidos CONCURRENTES a {args.url} ...")
    async with httpx.AsyncClient(base_url=args.url, timeout=args.timeout + 5) as client:
        t_inicio = time.monotonic()
        resultados = await asyncio.gather(
            *[_correr_un_pedido(client, q, not args.no_auto_approve, args.timeout) for q in queries]
        )
        duracion_total = time.monotonic() - t_inicio

    latencias = [r["latencia_total_s"] for r in resultados if r["status"] == "DONE"]
    exitosos = sum(1 for r in resultados if r["status"] == "DONE")
    fallidos = sum(1 for r in resultados if r["status"] == "FAILED")

    resumen = {
        "n_pedidos": args.n,
        "exitosos": exitosos,
        "fallidos": fallidos,
        "duracion_total_wall_clock_s": round(duracion_total, 3),
        "latencia_p50_s": round(_percentil(latencias, 50), 3),
        "latencia_p95_s": round(_percentil(latencias, 95), 3),
        "latencia_maxima_s": round(max(latencias), 3) if latencias else None,
        "resultados": resultados,
    }

    print("\n--- Resumen de la prueba de carga ---")
    print(f"Pedidos: {args.n} | OK: {exitosos} | FAILED: {fallidos}")
    print(f"Tiempo total (wall-clock, las {args.n} corriendo en paralelo): {resumen['duracion_total_wall_clock_s']}s")
    print(f"Latencia p50: {resumen['latencia_p50_s']}s | p95: {resumen['latencia_p95_s']}s | máxima: {resumen['latencia_maxima_s']}s")
    if fallidos:
        print(f"\n¡OJO! {fallidos} pedido(s) terminaron en FAILED -- revisá 'error' en el JSON de salida.")

    salida = Path(__file__).resolve().parent.parent / "screenshots" / "load_test_resultados.json"
    salida.parent.mkdir(exist_ok=True)
    salida.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultados guardados en {salida}")
    print(
        "\nAhora andá al dashboard de observabilidad (Phoenix: http://localhost:6006, o "
        "LangSmith) y sacá una captura con el costo por ejecución y la latencia p95 que "
        "muestra EL DASHBOARD para esta corrida -- guardala en screenshots/ (ver README)."
    )


if __name__ == "__main__":
    asyncio.run(main())
