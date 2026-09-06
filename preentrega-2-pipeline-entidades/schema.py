"""
Esquema Pydantic — el "contrato" de salida del pipeline.

Define qué forma debe tener el objeto que devuelve el LLM luego de leer un
párrafo de texto sin procesar (una descripción de arquitectura, un log de
error, un ticket de soporte, etc.).
"""

from enum import Enum
from typing import List

from pydantic import BaseModel, Field, field_validator


class NivelCriticidad(str, Enum):
    """Nivel de criticidad inferido a partir del texto analizado."""

    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class EntidadesTecnicas(BaseModel):
    """
    Entidades técnicas extraídas de un párrafo de texto sin procesar.

    Esta clase es el "contrato": cualquier salida del LLM que no cumpla
    estas restricciones debe fallar la validación (y disparar un reintento),
    en lugar de propagarse silenciosamente como un objeto inválido.
    """

    tecnologias: List[str] = Field(
        ...,
        description=(
            "Lista de tecnologías, lenguajes, frameworks, servicios cloud o "
            "herramientas mencionadas explícita o implícitamente en el texto."
        ),
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        ...,
        description=(
            "Nivel de criticidad del problema, incidente o arquitectura "
            "descrita, evaluando el impacto potencial en el negocio o la "
            "operación del sistema."
        ),
    )
    resumen_tecnico: str = Field(
        ...,
        description="Resumen técnico conciso (1 a 3 oraciones) del contenido analizado.",
    )

    @field_validator("tecnologias")
    @classmethod
    def tecnologias_no_vacia(cls, v: List[str]) -> List[str]:
        """La lista de tecnologías no puede estar vacía ni ser solo ruido."""
        limpio = [t.strip() for t in (v or []) if t and t.strip()]
        if not limpio:
            raise ValueError(
                "La lista de 'tecnologias' no puede estar vacía. Si el texto "
                "no nombra tecnologías de forma explícita, inferí las más "
                "probables a partir del contexto (por ejemplo, un stack trace "
                "de Java implica la JVM) en lugar de devolver una lista vacía."
            )
        return limpio

    @field_validator("resumen_tecnico")
    @classmethod
    def resumen_no_vacio(cls, v: str) -> str:
        """El resumen debe existir y tener contenido mínimamente útil."""
        v = (v or "").strip()
        if not v:
            raise ValueError("'resumen_tecnico' no puede estar vacío.")
        if len(v) < 10:
            raise ValueError(
                "'resumen_tecnico' es demasiado corto para ser un resumen "
                "técnico útil (mínimo 10 caracteres)."
            )
        return v
