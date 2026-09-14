"""
generation.py
=============

Paso de GENERACIÓN (opcional / bonus): el enunciado de esta pre-entrega no
pide un LLM de generación (solo ingesta + recuperador híbrido +
evaluación), pero se agrega acá para poder probar el pipeline de punta a
punta —pregunta real -> respuesta real— sin depender de una cuenta de
pago, y así tener evidencia concreta para pegar en el README.

Usa **Groq** como proveedor de LLM: corre modelos open-source (Llama) con
un free tier real, sin pedir tarjeta de crédito
(https://console.groq.com -> API Keys).

El prompt es un "filtro de veracidad" (mismo patrón que la pre-entrega 3):
responde solo con lo que aparece en los fragmentos recuperados por
`RAGSystem`, y dice explícitamente que no tiene esa información si no
está en el contexto, en vez de inventarla.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from config import GROQ_API_KEY, GROQ_MODEL, LLM_TEMPERATURE

logger = logging.getLogger("rag_generation")


@dataclass
class RespuestaGenerada:
    respuesta: str
    fuentes: list[str] = field(default_factory=list)


PROMPT_GENERACION = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sos un asistente técnico que responde preguntas ÚNICAMENTE con "
            "información que aparece en el CONTEXTO proporcionado (fragmentos "
            "recuperados por un sistema RAG híbrido, BM25 + búsqueda "
            "vectorial en Pinecone).\n\n"
            "Reglas estrictas:\n"
            "1. Si la respuesta está en el CONTEXTO, respondé de forma clara "
            "y concisa, basándote solo en lo que ahí dice.\n"
            "2. Si la respuesta NO está en el CONTEXTO, decí explícitamente "
            "que no tenés esa información en los documentos disponibles. "
            "NUNCA inventes ni completes con conocimiento externo.\n\n"
            "CONTEXTO:\n---\n{contexto}\n---",
        ),
        ("human", "{pregunta}"),
    ]
)


def _get_llm_generacion():
    if not GROQ_API_KEY:
        raise RuntimeError(
            "Falta GROQ_API_KEY. Conseguí una gratis en "
            "https://console.groq.com/keys y completala en tu .env."
        )
    from langchain_groq import ChatGroq

    return ChatGroq(model=GROQ_MODEL, temperature=LLM_TEMPERATURE)


def _formatear_contexto(documentos: list[Document]) -> str:
    if not documentos:
        return "(No se recuperó ningún fragmento relevante.)"

    bloques = []
    for doc in documentos:
        fuente_id = doc.metadata.get("fuente_id", doc.metadata.get("source", "desconocida"))
        bloques.append(f"[FUENTE: {fuente_id}]\n{doc.page_content}")
    return "\n\n---\n\n".join(bloques)


def _fuentes_de(documentos: list[Document]) -> list[str]:
    vistas: set[str] = set()
    ordenadas: list[str] = []
    for doc in documentos:
        fuente_id = doc.metadata.get("fuente_id", doc.metadata.get("source", "desconocida"))
        if fuente_id not in vistas:
            vistas.add(fuente_id)
            ordenadas.append(fuente_id)
    return ordenadas


def generar_respuesta(pregunta: str, documentos: list[Document]) -> RespuestaGenerada:
    """Genera una respuesta grounded (sync) a partir de los documentos ya
    recuperados por `RAGSystem.query()`."""
    llm = _get_llm_generacion()
    contexto = _formatear_contexto(documentos)
    mensajes = PROMPT_GENERACION.format_messages(contexto=contexto, pregunta=pregunta)

    logger.info("Generando respuesta con Groq (%s)...", GROQ_MODEL)
    respuesta_llm = llm.invoke(mensajes)

    return RespuestaGenerada(respuesta=respuesta_llm.content, fuentes=_fuentes_de(documentos))


async def agenerar_respuesta(pregunta: str, documentos: list[Document]) -> RespuestaGenerada:
    """Versión asíncrona de `generar_respuesta()`."""
    llm = _get_llm_generacion()
    contexto = _formatear_contexto(documentos)
    mensajes = PROMPT_GENERACION.format_messages(contexto=contexto, pregunta=pregunta)

    logger.info("Generando respuesta async con Groq (%s)...", GROQ_MODEL)
    respuesta_llm = await llm.ainvoke(mensajes)

    return RespuestaGenerada(respuesta=respuesta_llm.content, fuentes=_fuentes_de(documentos))
