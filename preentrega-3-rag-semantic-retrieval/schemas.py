"""
schemas.py
==========

Modelo Pydantic de salida para el sistema de recuperación semántica (RAG).

`RespuestaRAG` es el contrato que devuelve `get_rag_response()`: el texto de
la respuesta generada por el LLM (fundamentada exclusivamente en el
contexto recuperado) más las referencias a los fragmentos/documentos que se
usaron para construirla.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class RespuestaRAG(BaseModel):
    """Respuesta grounded (fundamentada en el contexto recuperado)."""

    respuesta: str = Field(
        ...,
        min_length=1,
        description=(
            "Respuesta a la pregunta del usuario, basada exclusivamente en "
            "el CONTEXTO provisto. Si la información no está en el "
            "contexto, este campo debe decir explícitamente que no se "
            "cuenta con esa información (nunca inventar una respuesta)."
        ),
    )
    fuentes: list[str] = Field(
        default_factory=list,
        description=(
            "Lista de identificadores de fuente (formato "
            "'archivo.md#fragmento_N') de los fragmentos del contexto "
            "efectivamente usados para construir la respuesta. Lista "
            "vacía si la respuesta fue 'no lo sé' porque no había "
            "contexto relevante."
        ),
    )

    @field_validator("respuesta")
    @classmethod
    def respuesta_no_vacia(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("La respuesta no puede estar vacía.")
        return value

    @field_validator("fuentes")
    @classmethod
    def fuentes_sin_duplicados(cls, value: list[str]) -> list[str]:
        vistos: set[str] = set()
        deduplicado: list[str] = []
        for fuente in value:
            fuente = fuente.strip()
            if fuente and fuente not in vistos:
                vistos.add(fuente)
                deduplicado.append(fuente)
        return deduplicado
