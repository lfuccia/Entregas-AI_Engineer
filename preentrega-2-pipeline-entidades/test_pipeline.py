"""
test_pipeline.py
================

Mini-script de prueba asíncrono para el Pipeline de Extracción de Entidades
Técnicas.

Ejecuta la cadena LCEL (`chain.process_text`) contra tres párrafos de
ejemplo:
    1. Un texto claro (arquitectura típica con tecnologías explícitas).
    2. Un log de error con criticidad alta.
    3. Un texto AMBIGUO (la "prueba de estrés" sugerida en el enunciado),
       para verificar que el validador reacciona con una excepción clara o
       que el modelo se recupera infiriendo tecnologías razonables.

Requiere tener configurada la API key del proveedor elegido (ver
`.env.example` / `LLM_PROVIDER` en `.env`).

Uso:
    python test_pipeline.py
"""

from __future__ import annotations

import asyncio
import json
import logging

from pydantic import ValidationError

from chain import logger, process_text

EJEMPLOS = {
    "arquitectura_clara": (
        "Nuestro backend expone una API en FastAPI que atiende picos de "
        "5000 requests por segundo. Usamos Redis como capa de caché para "
        "las consultas más frecuentes y PostgreSQL como base de datos "
        "principal para persistencia. En el último incidente detectamos "
        "que el pool de conexiones a PostgreSQL se agotaba bajo carga, "
        "generando timeouts intermitentes en producción."
    ),
    "log_error_critico": (
        "[2026-09-12 03:14:02] ERROR OutOfMemoryError en el servicio de "
        "pagos (Java 17, Spring Boot). El pod fue reiniciado por "
        "Kubernetes tras superar el límite de memoria. Se perdieron 12 "
        "transacciones en cola de Kafka que no llegaron a persistirse en "
        "MongoDB. Impacto: caída total del checkout durante 6 minutos."
    ),
    "texto_ambiguo": (
        "El sistema anduvo raro toda la mañana, algunos usuarios se "
        "quejaron de que 'todo iba lento' pero después se normalizó solo. "
        "No quedó claro si fue la red, el servidor o algo del lado del "
        "cliente. Nadie encontró nada raro en los dashboards."
    ),
}


async def _ejecutar_ejemplo(nombre: str, texto: str) -> None:
    separador = "=" * 70
    print(f"\n{separador}\nEJEMPLO: {nombre}\n{separador}")
    print(f"Texto de entrada:\n  {texto}\n")

    try:
        resultado = await process_text(texto)
    except ValidationError as exc:
        # El validador de Pydantic rechazó la respuesta final tras agotar
        # los reintentos: esto es exactamente lo que el ejercicio pide
        # verificar en la "prueba de estrés".
        print(f"❌ Validación falló (esperable en textos ambiguos): {exc}")
        return
    except Exception as exc:  # noqa: BLE001 - mini-script de demostración
        print(f"❌ El pipeline falló tras agotar los reintentos: {exc}")
        return

    print("✅ Objeto validado:")
    print(json.dumps(resultado.model_dump(mode="json"), indent=2, ensure_ascii=False))


async def main() -> None:
    logging.getLogger("rag_entity_pipeline").setLevel(logging.INFO)
    logger.info("Arrancando mini-script de prueba asíncrono...")

    for nombre, texto in EJEMPLOS.items():
        await _ejecutar_ejemplo(nombre, texto)

    print("\nListo. Revisá los logs de arriba para ver validaciones y reintentos.")


if __name__ == "__main__":
    asyncio.run(main())
