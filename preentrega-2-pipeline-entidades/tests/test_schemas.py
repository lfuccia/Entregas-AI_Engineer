"""
tests/test_schemas.py

Tests unitarios (offline, sin LLM) para el contrato Pydantic de
`schemas.py`. Verifican las restricciones mencionadas en el enunciado:
la lista de tecnologías no puede estar vacía, el enum de criticidad valida
sus valores, y el resumen técnico no puede estar en blanco.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas import EntidadesTecnicas, NivelCriticidad


def test_objeto_valido_se_construye_correctamente():
    obj = EntidadesTecnicas(
        tecnologias=["FastAPI", "Redis", "PostgreSQL"],
        nivel_de_criticidad="alta",
        resumen_tecnico="API con caché en Redis y persistencia en PostgreSQL.",
    )
    assert obj.tecnologias == ["FastAPI", "Redis", "PostgreSQL"]
    assert obj.nivel_de_criticidad is NivelCriticidad.ALTA
    assert obj.resumen_tecnico.startswith("API con caché")


def test_lista_de_tecnologias_vacia_lanza_excepcion():
    with pytest.raises(ValidationError):
        EntidadesTecnicas(
            tecnologias=[],
            nivel_de_criticidad="baja",
            resumen_tecnico="Texto de prueba suficientemente largo.",
        )


def test_lista_de_tecnologias_solo_blancos_lanza_excepcion():
    with pytest.raises(ValidationError):
        EntidadesTecnicas(
            tecnologias=["   ", ""],
            nivel_de_criticidad="baja",
            resumen_tecnico="Texto de prueba suficientemente largo.",
        )


def test_tecnologias_duplicadas_se_deduplican_case_insensitive():
    obj = EntidadesTecnicas(
        tecnologias=["Redis", "redis", "  Redis  ", "PostgreSQL"],
        nivel_de_criticidad="media",
        resumen_tecnico="Texto de prueba suficientemente largo.",
    )
    assert obj.tecnologias == ["Redis", "PostgreSQL"]


def test_nivel_de_criticidad_invalido_lanza_excepcion():
    with pytest.raises(ValidationError):
        EntidadesTecnicas(
            tecnologias=["FastAPI"],
            nivel_de_criticidad="urgente",  # no es un valor válido del enum
            resumen_tecnico="Texto de prueba suficientemente largo.",
        )


def test_resumen_tecnico_vacio_lanza_excepcion():
    with pytest.raises(ValidationError):
        EntidadesTecnicas(
            tecnologias=["FastAPI"],
            nivel_de_criticidad="baja",
            resumen_tecnico="   ",
        )


def test_resumen_tecnico_demasiado_corto_lanza_excepcion():
    with pytest.raises(ValidationError):
        EntidadesTecnicas(
            tecnologias=["FastAPI"],
            nivel_de_criticidad="baja",
            resumen_tecnico="corto",
        )
