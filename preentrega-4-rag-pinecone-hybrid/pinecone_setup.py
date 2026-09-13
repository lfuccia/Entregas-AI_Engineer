"""
pinecone_setup.py
=================

Script de inicialización de infraestructura: verifica si el índice de
Pinecone (`config.INDEX_NAME`) ya existe y lo crea en modo **Serverless**
si hace falta.

Chequea explícitamente el error más común de este tipo de sistemas
("Mismatch de Dimensiones"): si el índice ya existe pero fue creado con una
dimensión distinta a la que devuelve el modelo de embeddings configurado
(`config.EMBEDDING_DIMENSION`), corta con un error claro en vez de dejar
que Pinecone rechace cada upsert uno por uno más adelante.

Uso:
    python pinecone_setup.py
"""

from __future__ import annotations

import logging
import os
import time

from config import (
    EMBEDDING_DIMENSION,
    INDEX_NAME,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_METRIC,
    PINECONE_REGION,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("pinecone_setup")


class DimensionMismatchError(Exception):
    """El índice ya existe con una dimensión distinta a la esperada."""


def _get_cliente_pinecone():
    from pinecone import Pinecone

    if not PINECONE_API_KEY:
        raise RuntimeError(
            "Falta PINECONE_API_KEY. Completá tu .env (ver .env.example)."
        )
    return Pinecone(api_key=PINECONE_API_KEY)


def asegurar_indice(esperar_listo: bool = True):
    """
    Idempotente: si `INDEX_NAME` ya existe, valida su dimensión y lo
    devuelve tal cual. Si no existe, lo crea en modo Serverless con la
    dimensión/métrica configuradas.
    """
    from pinecone import ServerlessSpec

    pc = _get_cliente_pinecone()
    indices_existentes = {i["name"] for i in pc.list_indexes()}

    if INDEX_NAME in indices_existentes:
        descripcion = pc.describe_index(INDEX_NAME)
        dimension_actual = descripcion.dimension
        if dimension_actual != EMBEDDING_DIMENSION:
            raise DimensionMismatchError(
                f"El índice '{INDEX_NAME}' ya existe con dimension="
                f"{dimension_actual}, pero el modelo de embeddings "
                f"configurado espera dimension={EMBEDDING_DIMENSION}. "
                f"Esto es exactamente el 'mismatch de dimensiones' que "
                f"menciona el enunciado: o cambiás EMBEDDING_DIMENSION/"
                f"EMBEDDING_MODEL en tu .env para que coincidan con el "
                f"índice existente, o borrás el índice y lo recreás con "
                f"la dimensión correcta."
            )
        logger.info(
            "El índice '%s' ya existe (dimension=%d, metric=%s). No se "
            "crea de nuevo.",
            INDEX_NAME,
            dimension_actual,
            descripcion.metric,
        )
        return pc.Index(INDEX_NAME)

    logger.info(
        "Creando índice Serverless '%s' (dimension=%d, metric=%s, cloud=%s, "
        "region=%s)...",
        INDEX_NAME,
        EMBEDDING_DIMENSION,
        PINECONE_METRIC,
        PINECONE_CLOUD,
        PINECONE_REGION,
    )
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric=PINECONE_METRIC,
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )

    if esperar_listo:
        while not pc.describe_index(INDEX_NAME).status.get("ready", False):
            logger.info("Esperando a que el índice quede listo...")
            time.sleep(2)

    logger.info("Índice '%s' creado y listo.", INDEX_NAME)
    return pc.Index(INDEX_NAME)


if __name__ == "__main__":
    asegurar_indice()
