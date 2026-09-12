"""
schemas.py
==========

Contrato de datos (esquema Pydantic) para el Pipeline de Extracción de
Entidades Técnicas.

Este modelo es el "contrato" que se le pasa a `model.with_structured_output()`
en `chain.py`. LangChain lo traduce automáticamente a un JSON Schema / tool
schema que el LLM debe respetar, y Pydantic valida (o rechaza) la respuesta
del modelo contra estas reglas antes de que llegue al resto de la
aplicación.

Campos requeridos por el ejercicio:
    - tecnologias: list[str]           -> nunca vacía, sin duplicados.
    - nivel_de_criticidad: enum        -> "baja" | "media" | "alta".
    - resumen_tecnico: str             -> resumen breve, no vacío.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class NivelCriticidad(str, Enum):
    """Nivel de criticidad técnica detectado en el texto de entrada."""

    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class EntidadesTecnicas(BaseModel):
    """
    Objeto validado que devuelve el pipeline de extracción de entidades
    técnicas a partir de un párrafo de texto sin procesar (por ejemplo, una
    descripción de arquitectura de software o un log de error).
    """

    tecnologias: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Lista de tecnologías, frameworks, lenguajes, bases de datos, "
            "protocolos o servicios mencionados explícitamente en el texto "
            "(ej. 'FastAPI', 'Redis', 'PostgreSQL'). Nunca debe estar vacía: "
            "si el texto no menciona ninguna tecnología reconocible, "
            "el modelo debe inferir la más plausible a partir del contexto "
            "(por ejemplo, 'HTTP' para un log de un servidor web genérico)."
        ),
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        ...,
        description=(
            "Nivel de criticidad técnica implícito o explícito en el "
            "texto: 'baja' (informativo, sin impacto), 'media' (degradación "
            "parcial o riesgo moderado) o 'alta' (caída de servicio, "
            "pérdida de datos, o impacto crítico en producción)."
        ),
    )
    resumen_tecnico: str = Field(
        ...,
        min_length=10,
        description=(
            "Resumen técnico breve (1-2 oraciones, en español) del "
            "contenido del texto, orientado a un/a ingeniero/a de guardia "
            "que necesita entender el problema en segundos."
        ),
    )

    @field_validator("tecnologias")
    @classmethod
    def tecnologias_no_vacias_ni_duplicadas(cls, value: list[str]) -> list[str]:
        """Limpia espacios, descarta strings vacíos y elimina duplicados
        (case-insensitive) preservando el orden de aparición."""
        limpio = [t.strip() for t in value if t and t.strip()]
        if not limpio:
            raise ValueError(
                "La lista de tecnologías no puede estar vacía ni contener "
                "solo strings en blanco."
            )
        vistos: set[str] = set()
        deduplicado: list[str] = []
        for tecnologia in limpio:
            clave = tecnologia.lower()
            if clave not in vistos:
                vistos.add(clave)
                deduplicado.append(tecnologia)
        return deduplicado

    @field_validator("resumen_tecnico")
    @classmethod
    def resumen_no_vacio(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("El resumen técnico no puede estar vacío.")
        return value

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
                    "nivel_de_criticidad": "alta",
                    "resumen_tecnico": (
                        "API con caché en Redis y persistencia en "
                        "PostgreSQL; cuello de botella en conexiones "
                        "concurrentes."
                    ),
                }
            ]
        }
    }
