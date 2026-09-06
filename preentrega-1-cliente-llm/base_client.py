"""
Interfaz común (clase base abstracta) para cualquier proveedor de LLM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, List, Optional

from schemas import ChatMessage, ModelConfig, ModelResponse


class BaseLLMClient(ABC):
    """
    Todo cliente concreto (OpenAIClient, AnthropicClient, ...) implementa
    esta interfaz. Gracias a esto, el resto del programa (AsyncLLMManager,
    main.py) puede usar cualquier proveedor de forma intercambiable, sin
    conocer los detalles de su SDK.
    """

    def __init__(self, config: ModelConfig, api_key: Optional[str] = None) -> None:
        self.config = config
        self.api_key = api_key

    @abstractmethod
    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        """Genera una respuesta completa (no streaming)."""
        raise NotImplementedError

    @abstractmethod
    def stream(self, messages: List[ChatMessage]) -> AsyncIterator[str]:
        """Generador asíncrono que entrega fragmentos de texto conforme llegan."""
        raise NotImplementedError

    @abstractmethod
    async def aclose(self) -> None:
        """Libera los recursos del cliente HTTP subyacente."""
        raise NotImplementedError
