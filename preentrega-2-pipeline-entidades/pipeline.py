"""
Pipeline de Extracción de Entidades Técnicas.

Componentes:
    1. Esquema Pydantic       -> schema.EntidadesTecnicas
    2. Prompt Template        -> PROMPT_TEMPLATE (modular: texto_entrada + format_instructions)
    3. Cadena LCEL            -> build_structured_chain() / build_parser_chain()
    4. Lógica de resiliencia  -> .with_retry() sobre la cadena completa

Se ofrecen dos formas de armar la cadena:

  - build_structured_chain(): la PREFERIDA. Usa `model.with_structured_output(...)`,
    que en Anthropic/OpenAI se apoya en function/tool calling nativo del
    proveedor. El propio proveedor garantiza JSON válido contra el schema,
    por lo que los reintentos cubren sobre todo fallas de red o de
    validación de negocio (ej. nuestro validador de lista vacía).

  - build_parser_chain(): alternativa didáctica con `PydanticOutputParser`,
    útil para modelos sin function calling o para ver explícitamente cómo
    se recupera el pipeline ante un JSON mal formado o incompleto (ver
    test_pipeline.py, que ejercita este camino sin necesitar API key).
"""

from __future__ import annotations

import os

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import ValidationError

from schema import EntidadesTecnicas

# ---------------------------------------------------------------------------
# 2. Prompt Template modular
# ---------------------------------------------------------------------------
# Acepta dos variables: el texto de entrada y las instrucciones de formato.
# Las instrucciones de formato quedan como variable de template (no fijas)
# para poder reutilizar el mismo prompt tanto con PydanticOutputParser
# (que necesita el JSON schema en texto) como con with_structured_output
# (que no lo necesita, porque el schema viaja como tool/function).

SYSTEM_TEMPLATE = """Sos un analista técnico senior especializado en arquitecturas de \
software, infraestructura cloud y troubleshooting de incidentes.

Tu tarea es leer el texto que te provee el usuario (puede ser la descripción de una \
arquitectura, un log de error, un ticket de soporte, etc.) y extraer, de forma \
estructurada:

1. Todas las tecnologías, lenguajes, frameworks, servicios cloud o herramientas \
   mencionadas o claramente implicadas en el texto.
2. El nivel de criticidad del escenario descrito ('baja', 'media' o 'alta'), \
   evaluando el impacto potencial en el negocio o en la operación del sistema.
3. Un resumen técnico breve y preciso (1 a 3 oraciones).

Si el texto es ambiguo o no menciona tecnologías de forma explícita, inferí las más \
probables a partir del contexto (por ejemplo, un error de "connection pool \
exhausted" sugiere una base de datos relacional) en lugar de dejar la lista vacía.

{format_instructions}
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        ("human", "Texto a analizar:\n\n{texto_entrada}"),
    ]
)


def build_prompt(format_instructions: str = "") -> ChatPromptTemplate:
    """Fija (parcializa) las instrucciones de formato del prompt modular."""
    return PROMPT_TEMPLATE.partial(format_instructions=format_instructions)


_NO_JSON_NEEDED = (
    "(No hace falta que devuelvas JSON en el texto de la respuesta: la "
    "estructura de salida se aplica automáticamente mediante function calling.)"
)


# ---------------------------------------------------------------------------
# 3 + 4. Cadena LCEL con resiliencia (.with_retry)
# ---------------------------------------------------------------------------

def build_structured_chain(model: BaseChatModel, *, max_intentos: int = 3) -> Runnable:
    """
    Cadena preferida: Prompt + model.with_structured_output(EntidadesTecnicas).

    Requiere un modelo con soporte de tool/function calling (Anthropic, OpenAI).
    """
    prompt = build_prompt(format_instructions=_NO_JSON_NEEDED)
    structured_model = model.with_structured_output(EntidadesTecnicas)
    chain = prompt | structured_model
    return chain.with_retry(
        retry_if_exception_type=(ValidationError, OutputParserException, ValueError),
        wait_exponential_jitter=True,
        stop_after_attempt=max_intentos,
    )


def build_parser_chain(model: BaseChatModel, *, max_intentos: int = 3) -> Runnable:
    """
    Cadena alternativa: Prompt + LLM + PydanticOutputParser.

    Sirve para proveedores sin function calling y para demostrar de forma
    explícita la recuperación ante JSON mal formado o incompleto (el LLM
    devuelve texto libre y el parser es quien intenta interpretarlo).
    """
    parser = PydanticOutputParser(pydantic_object=EntidadesTecnicas)
    prompt = build_prompt(format_instructions=parser.get_format_instructions())
    chain = prompt | model | parser
    return chain.with_retry(
        retry_if_exception_type=(ValidationError, OutputParserException, ValueError),
        wait_exponential_jitter=True,
        stop_after_attempt=max_intentos,
    )


def get_model(temperature: float = 0.0) -> BaseChatModel:
    """
    Devuelve un modelo de chat real. Prioriza Anthropic (Claude); si no hay
    ANTHROPIC_API_KEY, intenta OpenAI. Pensado para demo_real.py.
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model="claude-sonnet-4-5", temperature=temperature)
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
    raise OSError(
        "No se encontró ANTHROPIC_API_KEY ni OPENAI_API_KEY en el entorno. "
        "Configurá una de las dos para ejecutar demo_real.py (ver README.md), "
        "o corré `python -m pytest test_pipeline.py` para validar la lógica "
        "de resiliencia sin necesitar ninguna API key."
    )


def run_pipeline(texto_entrada: str, chain: Runnable | None = None) -> EntidadesTecnicas:
    """Atajo de conveniencia: arma la cadena preferida si no se pasa una."""
    if chain is None:
        chain = build_structured_chain(get_model())
    return chain.invoke({"texto_entrada": texto_entrada})
