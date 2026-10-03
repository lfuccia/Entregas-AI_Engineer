"""
tools/analisis.py
==================

Herramientas del Agente de Análisis/Cómputo: análisis de sentimiento
(léxico, offline, sin ningún modelo ni API externa) y cálculo de métricas
numéricas. Deliberadamente simples y deterministas — el objetivo de esta
pre-entrega es la orquestación multi-agente, no un clasificador de
sentimiento sofisticado.
"""

from __future__ import annotations

import statistics
import unicodedata

from langchain_core.tools import tool

_PALABRAS_POSITIVAS = {
    "excelente", "recomendado", "increible", "increíble", "bueno", "buen",
    "solido", "sólido", "mejoras", "mejora", "genial", "encanto", "perfecto",
}
_PALABRAS_NEGATIVAS = {
    "decepcionante", "mediocre", "malo", "mala", "cuelga", "problema",
    "lento", "caro", "falla", "defectuoso",
}


def _sin_tildes(texto: str) -> str:
    normalizado = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in normalizado if not unicodedata.combining(c)).lower()


@tool
def analizar_sentimiento(textos: list[str]) -> dict:
    """Analiza el sentimiento de una lista de textos (por ejemplo, las
    reseñas que trajo `buscar_informacion`) mediante conteo de palabras
    clave positivas/negativas. Usala cuando te pidan evaluar la opinión o
    percepción general a partir de texto libre.

    Devuelve {"sentimiento_general": "positivo"|"neutro"|"negativo",
    "score_promedio": <float, entre -1 y 1>, "detalle": [{"texto": <str>,
    "score": <float>}, ...]}. Un score > 0.15 es positivo, < -0.15 es
    negativo, y en el medio es neutro.

    Args:
        textos: lista de textos a analizar (ej. las reseñas encontradas).
    """
    detalle = []
    for texto in textos:
        palabras = _sin_tildes(texto).replace(".", " ").replace(",", " ").split()
        positivas = sum(1 for p in palabras if p in _PALABRAS_POSITIVAS)
        negativas = sum(1 for p in palabras if p in _PALABRAS_NEGATIVAS)
        total = positivas + negativas
        score = 0.0 if total == 0 else (positivas - negativas) / total
        detalle.append({"texto": texto, "score": round(score, 2)})

    score_promedio = round(statistics.mean(d["score"] for d in detalle), 2) if detalle else 0.0
    if score_promedio > 0.15:
        sentimiento_general = "positivo"
    elif score_promedio < -0.15:
        sentimiento_general = "negativo"
    else:
        sentimiento_general = "neutro"

    return {
        "sentimiento_general": sentimiento_general,
        "score_promedio": score_promedio,
        "detalle": detalle,
    }


@tool
def calcular_metricas(numeros: list[float]) -> dict:
    """Calcula métricas numéricas básicas (cantidad, suma, promedio, mínimo
    y máximo) sobre una lista de números — por ejemplo, las calificaciones
    (1 a 5) de un conjunto de reseñas. Usala cuando te pidan un promedio,
    total u otra cifra concreta a partir de datos numéricos, en vez de
    calcularlo "de memoria": así el resultado es exacto y verificable.

    Devuelve {"cantidad": <int>, "suma": <float>, "promedio": <float>,
    "minimo": <float>, "maximo": <float>}, o {"error": "..."} si la lista
    viene vacía.

    Args:
        numeros: lista de valores numéricos (ej. calificaciones de 1 a 5).
    """
    if not numeros:
        return {"error": "La lista de números está vacía."}

    return {
        "cantidad": len(numeros),
        "suma": round(sum(numeros), 2),
        "promedio": round(statistics.mean(numeros), 2),
        "minimo": min(numeros),
        "maximo": max(numeros),
    }
