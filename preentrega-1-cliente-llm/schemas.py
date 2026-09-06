"""
Esquemas Pydantic: el contrato de mensajes, la configuración del modelo y
la estructura de respuesta unificada.

Se definen ACÁ, primero que nada, para no propagar diccionarios sueltos
("dicts anidados") entre el resto de los módulos: todo lo que entra y sale
del cliente pasa por uno de estos modelos.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field, field_validator


class Role(str, Enum):
    """Rol de un mensaje dentro de la conversación."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Provider(str, Enum):
    """Proveedores de LLM soportados."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ChatMessage(BaseModel):
    """Un mensaje individual de la conversación."""

    role: Role
    content: str = Field(..., min_length=1, description="Texto del mensaje.")


class ModelConfig(BaseModel):
    """
    Configuración de una llamada al modelo.

    Se valida acá, con Pydantic, para no dejar pasar parámetros inválidos
    (una temperatura de 5, un max_tokens negativo, etc.) hasta el SDK del
    proveedor -- que fallaría con un error mucho menos claro.
    """

    provider: Provider
    model: str = Field(..., min_length=1, description="Nombre del modelo, ej. 'gpt-4o-mini'.")
    temperature: float = Field(1.0, ge=0.0, le=2.0, description="Creatividad del modelo (0-2).")
    max_tokens: int = Field(1024, gt=0, le=200_000, description="Tokens máximos de la respuesta.")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    timeout: float = Field(60.0, gt=0, description="Timeout por request, en segundos.")
    max_retries: int = Field(2, ge=0, le=5, description="Reintentos ante errores transitorios.")

    @field_validator("model")
    @classmethod
    def model_no_vacio(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("'model' no puede estar vacío.")
        return v.strip()


class ModelResponse(BaseModel):
    """
    Respuesta unificada de cualquier proveedor.

    Nunca se propaga una excepción cruda desde el cliente: si algo falla,
    'error' queda seteado (y 'content' vacío) en lugar de romper el
    programa que invoca al cliente.
    """

    provider: Provider
    model: str
    content: str = ""
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None
