"""
evaluate.py
===========

Script de Evaluación: mide Precision@k y Recall@k del recuperador híbrido
contra un "Golden Set" de preguntas con documento fuente conocido.

Para cada pregunta del golden set (`golden_set.json`, pares
`{"pregunta": ..., "documento_id_esperado": ...}`):
    - Se ejecuta `RAGSystem.query(pregunta, k=EVAL_TOP_K)`.
    - Recall@k  (por pregunta) = 1 si el documento esperado aparece entre
      los k recuperados, 0 si no.
    - Precision@k (por pregunta) = proporción de los k fragmentos
      recuperados que efectivamente vienen del documento esperado (cuántos
      de los k resultados son "realmente útiles" para esa pregunta).

El resumen final promedia ambas métricas sobre todas las preguntas del
golden set e imprime un reporte en consola.

Uso:
    python evaluate.py
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from config import EVAL_TOP_K, GOLDEN_SET_PATH
from rag_system import RAGSystem

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag_evaluate")


@dataclass
class ResultadoPregunta:
    pregunta: str
    documentos_relevantes: list[str]
    fuentes_recuperadas: list[str] = field(default_factory=list)
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0


def cargar_golden_set(ruta: Path) -> list[dict]:
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el golden set '{ruta}'. Ver golden_set.json de "
            f"ejemplo en la raíz del proyecto."
        )
    entradas = json.loads(ruta.read_text(encoding="utf-8"))
    if not entradas:
        raise ValueError(f"El golden set '{ruta}' está vacío.")
    return entradas


def _documentos_relevantes_de(entrada: dict) -> list[str]:
    """Soporta tanto un único 'documento_id_esperado' (formato pedido por
    el enunciado) como una lista 'documentos_relevantes' (por si a futuro
    una pregunta tiene más de un documento fuente válido)."""
    if "documentos_relevantes" in entrada:
        return list(entrada["documentos_relevantes"])
    return [entrada["documento_id_esperado"]]


def calcular_recall_at_k(relevantes: list[str], recuperados: list[str]) -> float:
    """Recall@k = |relevantes ∩ recuperados| / |relevantes|.
    Con un solo documento relevante (caso típico de este golden set), esto
    es simplemente 1.0 si ese documento aparece entre los recuperados, 0.0
    si no."""
    relevantes_unicos = set(relevantes)
    encontrados = relevantes_unicos & set(recuperados)
    return len(encontrados) / len(relevantes_unicos)


def calcular_precision_at_k(relevantes: list[str], recuperados: list[str]) -> float:
    """Precision@k = (cantidad de los k recuperados que son relevantes) / k.
    Mide qué porcentaje de lo que trajo el recuperador es "realmente útil"
    para responder esa pregunta puntual."""
    if not recuperados:
        return 0.0
    relevantes_unicos = set(relevantes)
    utiles = sum(1 for fuente in recuperados if fuente in relevantes_unicos)
    return utiles / len(recuperados)


def evaluar(rag_system: RAGSystem | None = None, top_k: int = EVAL_TOP_K) -> list[ResultadoPregunta]:
    rag_system = rag_system or RAGSystem(top_k=top_k)
    golden_set = cargar_golden_set(GOLDEN_SET_PATH)

    resultados: list[ResultadoPregunta] = []
    for entrada in golden_set:
        pregunta = entrada["pregunta"]
        relevantes = _documentos_relevantes_de(entrada)

        documentos_recuperados = rag_system.query(pregunta, k=top_k)
        fuentes = [doc.metadata.get("source", "desconocida") for doc in documentos_recuperados]

        resultado = ResultadoPregunta(
            pregunta=pregunta,
            documentos_relevantes=relevantes,
            fuentes_recuperadas=fuentes,
            recall_at_k=calcular_recall_at_k(relevantes, fuentes),
            precision_at_k=calcular_precision_at_k(relevantes, fuentes),
        )
        resultados.append(resultado)

    return resultados


def imprimir_reporte(resultados: list[ResultadoPregunta], top_k: int = EVAL_TOP_K) -> None:
    print("\n" + "=" * 78)
    print(f"REPORTE DE EVALUACIÓN — Precision@{top_k} / Recall@{top_k}")
    print("=" * 78)

    for i, r in enumerate(resultados, start=1):
        print(f"\n[{i}] Pregunta: {r.pregunta}")
        print(f"    Documento(s) esperado(s): {r.documentos_relevantes}")
        print(f"    Fuentes recuperadas (top-{top_k}): {r.fuentes_recuperadas}")
        print(f"    Recall@{top_k}:    {r.recall_at_k:.2f}")
        print(f"    Precision@{top_k}: {r.precision_at_k:.2f}")

    n = len(resultados)
    recall_promedio = sum(r.recall_at_k for r in resultados) / n
    precision_promedio = sum(r.precision_at_k for r in resultados) / n

    print("\n" + "-" * 78)
    print("RESUMEN")
    print("-" * 78)
    print(f"Preguntas evaluadas: {n}")
    print(f"Recall@{top_k} promedio:    {recall_promedio:.2%}")
    print(f"Precision@{top_k} promedio: {precision_promedio:.2%}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    resultados = evaluar()
    imprimir_reporte(resultados)
