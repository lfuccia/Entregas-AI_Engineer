"""
rag_chain.py
============

Capa de Recuperación + Generación Grounded del sistema RAG.

Arma la cadena LCEL:

    {contexto, pregunta, instrucciones_formato} | PROMPT | LLM | PydanticOutputParser

- El **retriever** convierte la pregunta en un embedding (con el MISMO
  modelo usado en `ingest.py`, vía `embeddings.get_embeddings()`) y trae los
  `top_k` fragmentos más similares de ChromaDB (k entre 3 y 5: ver
  `RAG_TOP_K` — evita el "contexto infinito" / Lost in the Middle).
- El **prompt** actúa como filtro de veracidad: instruye al modelo a
  responder solo con lo que está en el CONTEXTO, y a decir explícitamente
  que no tiene esa información si no aparece ahí.
- El **parser** es un `PydanticOutputParser(RespuestaRAG)`, envuelto en un
  `OutputFixingParser` para tolerar JSON levemente mal formado (el mismo
  LLM se usa para corregir su propia salida antes de fallar).

Expone la función asíncrona `get_rag_response(query)`.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from ingest import ejecutar_ingesta
from schemas import RespuestaRAG

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag_chain")

# top_k entre 3 y 5 (evita el "contexto infinito" / Lost in the Middle)
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))


# ---------------------------------------------------------------------------
# Cliente LLM configurable (mismo patrón que la pre-entrega 2)
# ---------------------------------------------------------------------------
def _get_llm() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model_name = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
        return ChatAnthropic(model=model_name, temperature=temperature)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model_name, temperature=temperature)

    raise ValueError(
        f"LLM_PROVIDER desconocido: '{provider}'. Usá 'anthropic' u 'openai'."
    )


# ---------------------------------------------------------------------------
# Prompt de sistema: filtro de veracidad
# ---------------------------------------------------------------------------
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sos un asistente técnico que responde preguntas ÚNICAMENTE con "
            "información que aparece en el CONTEXTO proporcionado.\n\n"
            "Reglas estrictas (filtro de veracidad):\n"
            "1. Si la respuesta está en el CONTEXTO, respondé de forma clara "
            "y concisa, basándote solo en lo que ahí dice.\n"
            "2. Si la respuesta NO está en el CONTEXTO, o el contexto no "
            "alcanza para responder con certeza, decí explícitamente que no "
            "tenés acceso a esa información en los documentos disponibles. "
            "NUNCA inventes ni completes con conocimiento externo, aunque "
            "creas saber la respuesta.\n"
            "3. En el campo 'fuentes' listá SOLO los identificadores "
            "[FUENTE: ...] de los fragmentos del contexto que realmente "
            "usaste para responder. Si respondiste 'no lo sé', dejá "
            "'fuentes' como una lista vacía.\n\n"
            "CONTEXTO:\n"
            "---\n"
            "{contexto}\n"
            "---\n\n"
            "{instrucciones_formato}",
        ),
        ("human", "{pregunta}"),
    ]
)

_parser_base = PydanticOutputParser(pydantic_object=RespuestaRAG)


# ---------------------------------------------------------------------------
# Formateo de documentos recuperados -> bloque de contexto con [FUENTE: ...]
# ---------------------------------------------------------------------------
def _formatear_contexto(documentos: list[Document]) -> str:
    if not documentos:
        return "(No se recuperó ningún fragmento relevante de la base vectorial.)"

    bloques = []
    for doc in documentos:
        fuente_id = doc.metadata.get(
            "fuente_id", doc.metadata.get("source", "desconocida")
        )
        bloques.append(f"[FUENTE: {fuente_id}]\n{doc.page_content}")
    return "\n\n---\n\n".join(bloques)


# ---------------------------------------------------------------------------
# Cadena LCEL (lazy, memoizada — evita instanciar el vectorstore/LLM al importar)
# ---------------------------------------------------------------------------
_chain_cache = None
_retriever_cache = None


def _get_retriever():
    global _retriever_cache
    if _retriever_cache is None:
        vectorstore: Chroma = ejecutar_ingesta(force=False)
        _retriever_cache = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RAG_TOP_K},
        )
        logger.info("Retriever listo (top_k=%d).", RAG_TOP_K)
    return _retriever_cache


def get_chain():
    """Arma (una sola vez) la cadena LCEL completa: retriever -> prompt ->
    LLM -> PydanticOutputParser, con OutputFixingParser como red de
    seguridad ante JSON mal formado."""
    global _chain_cache
    if _chain_cache is not None:
        return _chain_cache

    from langchain.output_parsers import OutputFixingParser

    retriever = _get_retriever()
    llm = _get_llm()
    parser_robusto = OutputFixingParser.from_llm(parser=_parser_base, llm=llm)

    recuperar_y_formatear = retriever | RunnableLambda(_formatear_contexto)

    _chain_cache = (
        RunnableParallel(
            contexto=recuperar_y_formatear,
            pregunta=RunnablePassthrough(),
            instrucciones_formato=lambda _: _parser_base.get_format_instructions(),
        )
        | PROMPT
        | llm
        | parser_robusto
    )
    return _chain_cache


# ---------------------------------------------------------------------------
# Función pública asíncrona
# ---------------------------------------------------------------------------
async def get_rag_response(query: str) -> RespuestaRAG:
    """
    Ejecuta el flujo RAG completo de forma asíncrona:
        1. Recupera los `RAG_TOP_K` fragmentos más similares a `query` en
           ChromaDB (usando el mismo modelo de embeddings que la ingesta).
        2. Arma el prompt con esos fragmentos como CONTEXTO.
        3. Llama al LLM de forma asíncrona (`.ainvoke`).
        4. Parsea la respuesta a `RespuestaRAG` (texto + fuentes).
    """
    if not query or not query.strip():
        raise ValueError("La consulta no puede estar vacía.")

    logger.info("Consulta recibida: %r", query)
    chain = get_chain()

    resultado: RespuestaRAG = await chain.ainvoke(query)

    logger.info(
        "Respuesta generada (%d fuentes citadas): %s",
        len(resultado.fuentes),
        resultado.fuentes,
    )
    return resultado
