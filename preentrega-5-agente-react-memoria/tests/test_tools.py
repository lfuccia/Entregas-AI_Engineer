"""
tests/test_tools.py

Tests unitarios offline (sin LLM, sin red) de la lógica pura de las
herramientas: `buscar_cliente` y `buscar_pedidos`. Como son `@tool` de
LangChain, se invocan con `.invoke({...})` pasando los argumentos como
dict (o `.func(...)` para llamarlas como función Python común).
"""

from __future__ import annotations

from tools import buscar_cliente, buscar_pedidos


def test_buscar_cliente_por_id_exacto():
    resultado = buscar_cliente.invoke({"nombre_o_id": "102"})
    assert resultado == {"encontrado": True, "cliente_id": 102, "nombre": "Carlos Méndez"}


def test_buscar_cliente_por_id_inexistente():
    resultado = buscar_cliente.invoke({"nombre_o_id": "999"})
    assert resultado["encontrado"] is False
    assert "error" in resultado


def test_buscar_cliente_por_nombre_unico_ignora_tildes_y_mayusculas():
    resultado = buscar_cliente.invoke({"nombre_o_id": "martina gomez"})
    assert resultado == {"encontrado": True, "cliente_id": 103, "nombre": "Martina Gómez"}


def test_buscar_cliente_ambiguo_devuelve_candidatos():
    resultado = buscar_cliente.invoke({"nombre_o_id": "Gómez"})
    assert resultado["encontrado"] is False
    assert resultado["ambiguo"] is True
    ids = {c["cliente_id"] for c in resultado["candidatos"]}
    assert ids == {103, 104}


def test_buscar_cliente_sin_coincidencias():
    resultado = buscar_cliente.invoke({"nombre_o_id": "Nadie Existente"})
    assert resultado["encontrado"] is False
    assert "error" in resultado


def test_buscar_pedidos_cliente_valido():
    resultado = buscar_pedidos.invoke({"cliente_id": 102})
    assert resultado["cliente_id"] == 102
    assert resultado["pedidos"] == 3
    assert resultado["total"] == 14500
    assert resultado["ultimo_pedido"]["id"] == "P-2044"


def test_buscar_pedidos_cliente_sin_pedidos_no_es_error():
    resultado = buscar_pedidos.invoke({"cliente_id": 104})
    assert resultado["pedidos"] == 0
    assert resultado["total"] == 0
    assert "error" not in resultado


def test_buscar_pedidos_cliente_inexistente_devuelve_error():
    resultado = buscar_pedidos.invoke({"cliente_id": 999})
    assert "error" in resultado
