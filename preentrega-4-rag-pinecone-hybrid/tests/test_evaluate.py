"""
tests/test_evaluate.py

Tests unitarios offline para las métricas de `evaluate.py` (Precision@k,
Recall@k) y la carga del golden set. No requieren RAGSystem ni Pinecone:
son funciones puras sobre listas de IDs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluate import (
    _documentos_relevantes_de,
    calcular_precision_at_k,
    calcular_recall_at_k,
    cargar_golden_set,
)


def test_recall_1_0_cuando_el_relevante_esta_entre_los_recuperados():
    assert calcular_recall_at_k(["a.md"], ["x.md", "a.md", "y.md"]) == 1.0


def test_recall_0_0_cuando_el_relevante_no_aparece():
    assert calcular_recall_at_k(["a.md"], ["x.md", "y.md", "z.md"]) == 0.0


def test_recall_con_multiples_relevantes_es_proporcion_encontrada():
    # 1 de 2 documentos relevantes aparece entre los recuperados -> 0.5
    assert calcular_recall_at_k(["a.md", "b.md"], ["a.md", "x.md"]) == 0.5


def test_precision_cuenta_cuantos_de_los_recuperados_son_utiles():
    # 2 de los 4 recuperados son del documento relevante -> 0.5
    recuperados = ["a.md", "x.md", "a.md", "y.md"]
    assert calcular_precision_at_k(["a.md"], recuperados) == 0.5


def test_precision_es_cero_si_ninguno_es_relevante():
    assert calcular_precision_at_k(["a.md"], ["x.md", "y.md"]) == 0.0


def test_precision_con_lista_vacia_de_recuperados_es_cero():
    assert calcular_precision_at_k(["a.md"], []) == 0.0


def test_documentos_relevantes_de_soporta_formato_simple():
    entrada = {"pregunta": "¿?", "documento_id_esperado": "a.md"}
    assert _documentos_relevantes_de(entrada) == ["a.md"]


def test_documentos_relevantes_de_soporta_lista_explicita():
    entrada = {"pregunta": "¿?", "documentos_relevantes": ["a.md", "b.md"]}
    assert _documentos_relevantes_de(entrada) == ["a.md", "b.md"]


def test_cargar_golden_set_real_tiene_5_preguntas():
    ruta = Path(__file__).resolve().parent.parent / "golden_set.json"
    golden_set = cargar_golden_set(ruta)
    assert len(golden_set) == 5
    for entrada in golden_set:
        assert "pregunta" in entrada
        assert "documento_id_esperado" in entrada


def test_cargar_golden_set_inexistente_lanza_excepcion(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        cargar_golden_set(tmp_path / "no_existe.json")


def test_cargar_golden_set_vacio_lanza_excepcion(tmp_path: Path):
    ruta = tmp_path / "vacio.json"
    ruta.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(ValueError):
        cargar_golden_set(ruta)
