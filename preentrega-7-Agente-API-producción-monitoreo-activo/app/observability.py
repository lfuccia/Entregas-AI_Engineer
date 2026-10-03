"""
observability.py
=================

Inicializa el tracing/observabilidad ACTIVA del sistema multi-agente:
cada llamada al LLM (supervisor, investigador, analista) y cada paso del
grafo de LangGraph queda instrumentado automáticamente — sin tener que
agregar decoradores manuales en cada nodo — vía `OpenInference`
(el estándar de instrumentación de OpenTelemetry para apps de LLM).

Dos proveedores soportados (elegís con `OBSERVABILITY_PROVIDER` en
`.env`):

- "phoenix" (default): Arize Phoenix. 100% gratis, open-source y
  self-hosted — no pide API key ni cuenta. Se levanta con
  `docker compose up` (ver `docker-compose.yml`) o
  `python -m phoenix.server.main serve`, y el dashboard queda en
  http://localhost:6006. Esta app solo necesita el paquete liviano
  `arize-phoenix-otel` (el exportador OTLP), NO el server completo — el
  server corre aparte, en su propio contenedor/proceso.
- "langsmith": LangSmith de LangChain. Tiene plan gratuito pero requiere
  registrarse y una API key (`LANGCHAIN_API_KEY`). Al estar basado
  pura y exclusivamente en variables de entorno estándar de LangChain
  (`LANGCHAIN_TRACING_V2`, etc.), NO hace falta tocar código: alcanza
  con fijar esas variables de entorno antes de construir el grafo.
- "none": no instrumenta nada (para tests/CI, para no exigir ninguna de
  las dos integraciones).

`init_observability()` se llama UNA sola vez, en el lifespan de la app
de FastAPI (`app/main.py`), antes de construir el grafo.
"""

from __future__ import annotations

import logging
import os

from app.config import (
    LANGCHAIN_API_KEY,
    LANGCHAIN_PROJECT,
    OBSERVABILITY_PROVIDER,
    PHOENIX_COLLECTOR_ENDPOINT,
    PHOENIX_PROJECT_NAME,
)

logger = logging.getLogger(__name__)

_inicializado = False


def init_observability() -> str:
    """Inicializa el proveedor de tracing configurado. Devuelve un string
    describiendo qué quedó activo (para loguearlo/mostrarlo en el
    arranque de la API). Es idempotente: llamarlo más de una vez (p.ej.
    en tests que recargan la app) no vuelve a instrumentar dos veces."""
    global _inicializado
    if _inicializado:
        return f"observabilidad ya inicializada ({OBSERVABILITY_PROVIDER})"

    if OBSERVABILITY_PROVIDER == "phoenix":
        resultado = _init_phoenix()
    elif OBSERVABILITY_PROVIDER == "langsmith":
        resultado = _init_langsmith()
    elif OBSERVABILITY_PROVIDER == "none":
        resultado = "observabilidad desactivada (OBSERVABILITY_PROVIDER=none)"
    else:
        resultado = (
            f"OBSERVABILITY_PROVIDER='{OBSERVABILITY_PROVIDER}' no reconocido "
            "(usar 'phoenix', 'langsmith' o 'none'); no se instrumentó nada."
        )
        logger.warning(resultado)

    _inicializado = True
    logger.info(resultado)
    return resultado


def _init_phoenix() -> str:
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register

        tracer_provider = register(
            project_name=PHOENIX_PROJECT_NAME,
            endpoint=PHOENIX_COLLECTOR_ENDPOINT,
            batch=True,
            set_global_tracer_provider=True,
            verbose=False,
        )
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        return (
            "tracing con Arize Phoenix activo -> exportando a "
            f"{PHOENIX_COLLECTOR_ENDPOINT} (proyecto '{PHOENIX_PROJECT_NAME}'). "
            "Dashboard en http://localhost:6006 (si Phoenix está levantado con "
            "`docker compose up`)."
        )
    except Exception as exc:  # pragma: no cover - defensivo, no debe tumbar la API
        return (
            "no se pudo inicializar el tracing de Phoenix "
            f"({exc.__class__.__name__}: {exc}). La API sigue funcionando "
            "igual, simplemente sin trazas. Revisá que Phoenix esté levantado "
            "en PHOENIX_COLLECTOR_ENDPOINT."
        )


def _init_langsmith() -> str:
    if not LANGCHAIN_API_KEY:
        return (
            "OBSERVABILITY_PROVIDER=langsmith pero falta LANGCHAIN_API_KEY en "
            "el .env; no se activó el tracing. Conseguí una key gratis en "
            "https://smith.langchain.com y completá el .env."
        )
    # LangSmith se activa pura y exclusivamente con variables de entorno
    # estándar de LangChain -- no hace falta instrumentar nada a mano, el
    # propio `langchain_core` manda las trazas si estas variables están
    # seteadas ANTES de construir los LLMs/el grafo.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    return f"tracing con LangSmith activo -> proyecto '{LANGCHAIN_PROJECT}' en https://smith.langchain.com"
