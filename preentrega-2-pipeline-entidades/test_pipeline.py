"""
Prueba de estrés del pipeline — 100% offline, sin necesitar API key.

Usa `FakeListChatModel` para simular las respuestas de un LLM y así poder
verificar de forma determinística:

  1. El "camino feliz": JSON bien formado en el primer intento.
  2. Recuperación ante JSON mal formado (sintaxis inválida) en el primer
     intento, corregido en el segundo -> demuestra que `.with_retry()`
     funciona.
  3. Recuperación ante JSON incompleto (falta un campo requerido) en el
     primer intento, corregido en el segundo.
  4. Un caso ambiguo/adversarial donde el modelo *insiste* en devolver una
     lista de tecnologías vacía en todos los intentos: el validador de
     Pydantic debe seguir rechazando la respuesta y, agotados los
     reintentos, el pipeline debe propagar la excepción en lugar de
     devolver un objeto inválido.

Correr con:  python -m pytest test_pipeline.py -v
"""

from __future__ import annotations

import json

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from pipeline import build_parser_chain

TEXTO_EJEMPLO = (
    "Estamos viendo timeouts intermitentes en el servicio de checkout. El log "
    "muestra 'connection pool exhausted' contra la base Postgres, y el API "
    "Gateway en AWS devuelve 504 luego de 30 segundos. El equipo sospecha que "
    "el último deploy con Kubernetes cambió los límites de conexiones del "
    "pool de PgBouncer."
)

JSON_VALIDO = json.dumps(
    {
        "tecnologias": ["PostgreSQL", "AWS API Gateway", "Kubernetes", "PgBouncer"],
        "nivel_de_criticidad": "alta",
        "resumen_tecnico": (
            "Timeouts intermitentes en checkout por agotamiento del pool de "
            "conexiones de Postgres tras un cambio de configuración en el "
            "último deploy de Kubernetes."
        ),
    }
)

JSON_MAL_FORMADO = '{"tecnologias": ["PostgreSQL", "AWS"], "nivel_de_criticidad": "alta", '  # corte abrupto, JSON inválido

JSON_INCOMPLETO = json.dumps(
    {
        "tecnologias": ["PostgreSQL", "AWS"],
        # falta 'nivel_de_criticidad' y 'resumen_tecnico'
    }
)

JSON_LISTA_VACIA = json.dumps(
    {
        "tecnologias": [],
        "nivel_de_criticidad": "baja",
        "resumen_tecnico": "El texto no menciona ninguna tecnología concreta.",
    }
)


def _chain_con_respuestas(respuestas: list[str], max_intentos: int = 3):
    fake_model = FakeListChatModel(responses=respuestas)
    return build_parser_chain(fake_model, max_intentos=max_intentos)


def test_camino_feliz_primer_intento():
    """JSON válido desde el primer intento: no debería hacer falta reintentar."""
    chain = _chain_con_respuestas([JSON_VALIDO])
    resultado = chain.invoke({"texto_entrada": TEXTO_EJEMPLO})

    assert resultado.nivel_de_criticidad.value == "alta"
    assert "PostgreSQL" in resultado.tecnologias
    assert len(resultado.resumen_tecnico) >= 10


def test_recupera_de_json_mal_formado():
    """
    Primer intento: JSON con sintaxis inválida (corte abrupto).
    Segundo intento: JSON válido.
    El pipeline debe recuperarse gracias a .with_retry().
    """
    chain = _chain_con_respuestas([JSON_MAL_FORMADO, JSON_VALIDO])
    resultado = chain.invoke({"texto_entrada": TEXTO_EJEMPLO})

    assert resultado.nivel_de_criticidad.value == "alta"
    assert "Kubernetes" in resultado.tecnologias


def test_recupera_de_json_incompleto():
    """
    Primer intento: JSON válido como sintaxis pero incompleto (faltan campos
    requeridos del esquema). Segundo intento: JSON completo.
    """
    chain = _chain_con_respuestas([JSON_INCOMPLETO, JSON_VALIDO])
    resultado = chain.invoke({"texto_entrada": TEXTO_EJEMPLO})

    assert resultado.nivel_de_criticidad.value == "alta"
    assert resultado.resumen_tecnico.startswith("Timeouts")


def test_falla_de_forma_controlada_si_insiste_en_lista_vacia():
    """
    Caso adversarial: el modelo devuelve, en TODOS los intentos, una lista de
    tecnologías vacía (JSON válido, pero que viola nuestra regla de negocio).

    El validador de Pydantic debe seguir rechazando la respuesta; agotados
    los `max_intentos`, la cadena debe propagar la excepción en lugar de
    devolver silenciosamente un objeto inválido.
    """
    chain = _chain_con_respuestas([JSON_LISTA_VACIA], max_intentos=3)

    with pytest.raises(OutputParserException) as exc_info:
        chain.invoke({"texto_entrada": "Todo funciona bien, sin más detalle."})

    assert "tecnologias" in str(exc_info.value)


def test_agota_reintentos_configurados():
    """
    Si el JSON mal formado se repite más veces de las que permite
    `max_intentos`, la cadena también debe fallar (no reintenta infinito).
    """
    chain = _chain_con_respuestas([JSON_MAL_FORMADO], max_intentos=2)

    with pytest.raises(OutputParserException):
        chain.invoke({"texto_entrada": TEXTO_EJEMPLO})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
