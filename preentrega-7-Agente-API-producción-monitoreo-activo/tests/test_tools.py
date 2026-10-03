"""
tests/test_tools.py

Tests offline (sin LLM, sin red) de las herramientas propias.
"""

from __future__ import annotations

from app.tools.analisis import analizar_sentimiento, calcular_metricas
from app.tools.busqueda import buscar_informacion


def test_buscar_informacion_simulado_devuelve_resultados():
    resultado = buscar_informacion.invoke({"query": "Aurora X2"})
    assert resultado["proveedor"] == "simulado"
    assert len(resultado["resultados"]) >= 1
    assert all("texto" in r and "calificacion" in r for r in resultado["resultados"])


def test_buscar_informacion_sin_palabras_clave_devuelve_todo_el_corpus():
    resultado = buscar_informacion.invoke({"query": "a"})
    assert len(resultado["resultados"]) == 5


def test_analizar_sentimiento_detecta_positivo_y_negativo():
    resultado = analizar_sentimiento.invoke(
        {"textos": ["Producto excelente y recomendado.", "Producto decepcionante y defectuoso."]}
    )
    assert resultado["detalle"][0]["score"] > 0
    assert resultado["detalle"][1]["score"] < 0


def test_analizar_sentimiento_general_positivo():
    resultado = analizar_sentimiento.invoke(
        {"textos": ["Excelente, recomendado, sólido.", "Buen producto, mejoras notables."]}
    )
    assert resultado["sentimiento_general"] == "positivo"


def test_calcular_metricas_promedio_correcto():
    resultado = calcular_metricas.invoke({"numeros": [5, 3, 2, 5, 4]})
    assert resultado["cantidad"] == 5
    assert resultado["suma"] == 19
    assert resultado["promedio"] == 3.8
    assert resultado["minimo"] == 2
    assert resultado["maximo"] == 5


def test_calcular_metricas_lista_vacia_es_error_no_excepcion():
    resultado = calcular_metricas.invoke({"numeros": []})
    assert "error" in resultado
