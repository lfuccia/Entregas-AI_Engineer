"""
chain.py
========

Cadena LCEL (LangChain Expression Language) para el Pipeline de Extracción
de Entidades Técnicas.

Componentes que arma este módulo:
    1. Cliente LLM configurable (ChatOpenAI o ChatAnthropic) vía variable de
       entorno `LLM_PROVIDER`.
    2. `ChatPromptTemplate` modular: no usa f-strings hardcodeadas, sino
       variables de plantilla (`{texto_entrada}`, `{instrucciones_formato}`)
       que LangChain gestiona.
    3. Cadena LCEL:  prompt | model.with_structured_output(Schema)
    4. Lógica de resiliencia:
        - Detección explícita de `finish_reason` truncado (JSON incompleto
          por corte de tokens).
        - Detección de errores de parseo/validación (JSON mal formado).
        - Reintento automático con `.with_retry()`.
    5. Función asíncrona `process_text()` que ejecuta la cadena con
       `.ainvoke()` y deja logs de todo el proceso.

Uso rápido:
    >>> import asyncio
    >>> from chain import process_text
    >>> asyncio.run(process_text("Migramos el backend a FastAPI..."))
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ValidationError

from schemas import EntidadesTecnicas

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("rag_entity_pipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())


# ---------------------------------------------------------------------------
# Excepciones propias de resiliencia
# ---------------------------------------------------------------------------
class PipelineResilienciaError(Exception):
    """Clase base para errores que SÍ deben disparar un reintento."""


class RespuestaIncompletaError(PipelineResilienciaError):
    """El LLM cortó la respuesta antes de terminar el JSON (finish_reason
    de tipo 'length' / 'max_tokens')."""


class RespuestaMalFormadaError(PipelineResilienciaError):
    """El LLM devolvió un JSON que no parsea o no cumple el esquema
    Pydantic (`EntidadesTecnicas`)."""


# ---------------------------------------------------------------------------
# 1. Cliente LLM configurable (Módulo 1 reutilizado / generalizado)
# ---------------------------------------------------------------------------
def _get_llm() -> BaseChatModel:
    """
    Construye el cliente de chat según la variable de entorno
    `LLM_PROVIDER` ("anthropic" u "openai"). Reutiliza la misma lógica de
    configuración de cliente que el Módulo 1 (API key vía entorno,
    temperatura baja para tareas de extracción determinísticas).
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model_name = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
        logger.debug("Inicializando ChatAnthropic (%s)", model_name)
        return ChatAnthropic(model=model_name, temperature=temperature)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        logger.debug("Inicializando ChatOpenAI (%s)", model_name)
        return ChatOpenAI(model=model_name, temperature=temperature)

    raise ValueError(
        f"LLM_PROVIDER desconocido: '{provider}'. Usá 'anthropic' u 'openai'."
    )


# ---------------------------------------------------------------------------
# 2. Prompt Template modular (sin f-strings hardcodeadas)
# ---------------------------------------------------------------------------
# Se usa PydanticOutputParser únicamente para generar las instrucciones de
# formato que se inyectan como variable en el prompt (refuerzo textual del
# schema). El parseo real de la respuesta lo hace `.with_structured_output()`
# (método preferido para OpenAI/Anthropic, vía tool-calling nativo).
_parser_referencia = PydanticOutputParser(pydantic_object=EntidadesTecnicas)

PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sos un analista técnico senior especializado en arquitectura de "
            "software y SRE. Tu tarea es leer un párrafo de texto sin "
            "procesar (puede ser una descripción de arquitectura, un log de "
            "error, un ticket de incidente, etc.) y extraer, de forma "
            "estructurada, las entidades técnicas relevantes.\n\n"
            "Reglas:\n"
            "- Identificá TODAS las tecnologías, frameworks, lenguajes, "
            "bases de datos, protocolos o servicios mencionados o "
            "claramente implícitos en el texto.\n"
            "- Si el texto es ambiguo o no menciona tecnologías explícitas, "
            "inferí la(s) más plausible(s) a partir del contexto en lugar "
            "de dejar la lista vacía.\n"
            "- Asigná un nivel de criticidad ('baja', 'media', 'alta') "
            "según el impacto descripto (o implícito) en el texto.\n"
            "- Escribí un resumen técnico breve y accionable, en español.\n\n"
            "{instrucciones_formato}",
        ),
        (
            "human",
            "Texto de entrada a analizar:\n"
            "---\n"
            "{texto_entrada}\n"
            "---",
        ),
    ]
)


# ---------------------------------------------------------------------------
# 4. Lógica de resiliencia: validar la respuesta cruda ANTES de confiar en ella
# ---------------------------------------------------------------------------
def _validar_respuesta_cruda(resultado: dict) -> EntidadesTecnicas:
    """
    Punto de control de resiliencia de la cadena.

    Al usar `model.with_structured_output(Schema, include_raw=True)`,
    LangChain devuelve un dict con tres claves:
        - "raw":  el AIMessage crudo (incluye `response_metadata`, con el
                  `finish_reason`/`stop_reason` que reportó el proveedor).
        - "parsed": la instancia de `EntidadesTecnicas` ya validada, o
                  `None` si el parseo/validación falló.
        - "parsing_error": la excepción de parseo/validación, si la hubo.

    Acá evitamos el error común de "ignorar el finish_reason": si el
    proveedor cortó la respuesta por límite de tokens, el JSON puede
    parecer válido pero estar truncado/incompleto, así que lo tratamos
    como fallo explícito y forzamos un reintento (junto con cualquier
    JSON mal formado que Pydantic ya haya rechazado).
    """
    raw_message = resultado.get("raw")
    parsed = resultado.get("parsed")
    parsing_error = resultado.get("parsing_error")

    finish_reason = None
    if raw_message is not None:
        metadata = getattr(raw_message, "response_metadata", {}) or {}
        finish_reason = metadata.get("finish_reason") or metadata.get("stop_reason")

    if finish_reason in {"length", "max_tokens", "max_output_tokens"}:
        logger.warning(
            "Respuesta truncada por límite de tokens (finish_reason=%s). "
            "Se descarta y se reintenta.",
            finish_reason,
        )
        raise RespuestaIncompletaError(
            f"El LLM cortó la respuesta por límite de tokens "
            f"(finish_reason={finish_reason})."
        )

    if parsing_error is not None or parsed is None:
        logger.warning(
            "JSON mal formado o incompleto recibido del LLM (%s). "
            "Se descarta y se reintenta.",
            parsing_error,
        )
        raise RespuestaMalFormadaError(
            f"El LLM devolvió un JSON mal formado o incompleto: {parsing_error}"
        )

    logger.info(
        "Respuesta validada correctamente contra el esquema Pydantic: "
        "tecnologias=%s | criticidad=%s",
        parsed.tecnologias,
        parsed.nivel_de_criticidad.value,
    )
    return parsed


# ---------------------------------------------------------------------------
# 3. Ensamblado de la cadena LCEL + 4. Reintento (.with_retry())
# ---------------------------------------------------------------------------
MAX_REINTENTOS = int(os.getenv("PIPELINE_MAX_REINTENTOS", "3"))

_chain_cache: Runnable | None = None


def _construir_chain() -> Runnable:
    """
    Arma la cadena LCEL principal:

        chain = prompt | model.with_structured_output(Schema)

    envuelta con `.with_retry()` para tolerar JSON mal formado, respuestas
    incompletas (finish_reason) o errores transitorios del proveedor.
    """
    llm = _get_llm()

    # --- La línea clave que pide el ejercicio ---
    modelo_estructurado = llm.with_structured_output(
        EntidadesTecnicas,
        include_raw=True,  # necesario para poder inspeccionar finish_reason
    )
    cadena_base = PROMPT | modelo_estructurado | RunnableLambda(_validar_respuesta_cruda)
    # ---------------------------------------------

    return cadena_base.with_retry(
        retry_if_exception_type=(
            RespuestaIncompletaError,
            RespuestaMalFormadaError,
            ValidationError,
        ),
        wait_exponential_jitter=True,
        stop_after_attempt=MAX_REINTENTOS,
    )


def get_chain() -> Runnable:
    """Devuelve la cadena LCEL, construyéndola de forma perezosa (lazy) y
    cacheándola. Perezoso a propósito: así importar este módulo no requiere
    tener una API key configurada (útil para tests unitarios offline)."""
    global _chain_cache
    if _chain_cache is None:
        _chain_cache = _construir_chain()
    return _chain_cache


# ---------------------------------------------------------------------------
# 5. Función asíncrona pública: process_text()
# ---------------------------------------------------------------------------
async def process_text(text: str) -> EntidadesTecnicas:
    """
    Ejecuta el pipeline completo de forma asíncrona sobre un párrafo de
    texto sin procesar y devuelve un objeto `EntidadesTecnicas` validado.

    Levanta la última excepción si, tras `MAX_REINTENTOS` intentos, el LLM
    no logra devolver una respuesta completa y válida.
    """
    if not text or not text.strip():
        raise ValueError("El texto de entrada no puede estar vacío.")

    logger.info("Iniciando procesamiento (%d caracteres de entrada)...", len(text))
    chain = get_chain()

    try:
        resultado: EntidadesTecnicas = await chain.ainvoke(
            {
                "texto_entrada": text,
                "instrucciones_formato": _parser_referencia.get_format_instructions(),
            }
        )
    except Exception:
        logger.exception(
            "El pipeline agotó los %d reintentos sin obtener una respuesta "
            "válida.",
            MAX_REINTENTOS,
        )
        raise

    logger.info("Procesamiento finalizado con éxito.")
    return resultado
