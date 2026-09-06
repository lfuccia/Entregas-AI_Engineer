"""
AsyncLLMManager: punto de entrada único del pipeline.

Elige el proveedor concreto según `config.provider` y agrega, por encima de
cualquier cliente, la lógica de reintentos y manejo de errores: nada de lo
que pase acá debería poder tirar abajo el loop principal del programa.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Dict, List, Optional, Type

from anthropic_client import AnthropicClient
from base_client import BaseLLMClient
from exceptions import LLMFatalError, LLMStreamInterrupted, LLMTransientError
from openai_client import OpenAIClient
from schemas import ChatMessage, ModelConfig, ModelResponse, Provider

logger = logging.getLogger(__name__)

_CLIENTS: Dict[Provider, Type[BaseLLMClient]] = {
    Provider.OPENAI: OpenAIClient,
    Provider.ANTHROPIC: AnthropicClient,
}


def _backoff_seconds(attempt: int) -> float:
    """Backoff exponencial simple: 0.5s, 1s, 2s, 4s, ..."""
    return 0.5 * (2 ** (attempt - 1))


class AsyncLLMManager:
    """
    Fachada única para hablarle a cualquier proveedor soportado.

    Instancia el cliente concreto (`OpenAIClient` / `AnthropicClient`) según
    `config.provider`, y expone `generate()` / `stream()` con la misma
    firma sin importar qué proveedor haya detrás.
    """

    def __init__(self, config: ModelConfig, api_key: Optional[str] = None) -> None:
        self.config = config
        try:
            client_cls = _CLIENTS[config.provider]
        except KeyError as exc:  # pragma: no cover - Pydantic ya valida el enum
            raise LLMFatalError(f"Proveedor no soportado: {config.provider}") from exc
        self._client: BaseLLMClient = client_cls(config, api_key=api_key)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        """
        Genera una respuesta completa.

        Reintenta ante errores transitorios (con backoff exponencial) hasta
        `config.max_retries` veces. Ante un error fatal, o agotados los
        reintentos, devuelve un `ModelResponse` con `error` seteado en vez
        de dejar propagar la excepción: quien llama a `generate()` nunca
        necesita un `try/except` para no crashear.
        """
        attempts = self.config.max_retries + 1
        last_error = "Error desconocido."

        for attempt in range(1, attempts + 1):
            try:
                return await self._client.generate(messages)
            except LLMFatalError as exc:
                logger.warning("Error fatal de %s (sin reintentar): %s", self.config.provider.value, exc)
                return self._error_response(str(exc))
            except LLMTransientError as exc:
                last_error = str(exc)
                logger.warning(
                    "Intento %s/%s falló (transitorio) en %s: %s",
                    attempt, attempts, self.config.provider.value, exc,
                )
                if attempt < attempts:
                    await asyncio.sleep(_backoff_seconds(attempt))
            except Exception as exc:  # noqa: BLE001 - último resguardo: nunca crashear
                logger.exception("Error inesperado en generate(): %s", exc)
                return self._error_response(f"Error inesperado: {exc}")

        return self._error_response(last_error)

    async def stream(self, messages: List[ChatMessage]) -> AsyncIterator[str]:
        """
        Entrega fragmentos de texto conforme llegan del proveedor.

        Reintenta la apertura del stream ante errores transitorios. Si el
        error ocurre después de haber emitido contenido (no se puede
        "deshacer" texto ya entregado) o es fatal, en vez de propagar la
        excepción se emite un último fragmento con el error y se corta:
        el `async for` de quien consume el stream nunca explota.
        """
        attempts = self.config.max_retries + 1

        for attempt in range(1, attempts + 1):
            try:
                async for chunk in self._client.stream(messages):
                    yield chunk
                return
            except LLMFatalError as exc:
                logger.warning("Error fatal de %s (sin reintentar): %s", self.config.provider.value, exc)
                yield self._error_marker(str(exc))
                return
            except LLMStreamInterrupted as exc:
                logger.warning("Stream interrumpido en %s: %s", self.config.provider.value, exc)
                yield self._error_marker(str(exc))
                return
            except LLMTransientError as exc:
                logger.warning(
                    "Intento %s/%s de streaming falló (transitorio) en %s: %s",
                    attempt, attempts, self.config.provider.value, exc,
                )
                if attempt >= attempts:
                    yield self._error_marker(str(exc))
                    return
                await asyncio.sleep(_backoff_seconds(attempt))
            except Exception as exc:  # noqa: BLE001 - último resguardo: nunca crashear
                logger.exception("Error inesperado en stream(): %s", exc)
                yield self._error_marker(f"Error inesperado: {exc}")
                return

    def _error_response(self, message: str) -> ModelResponse:
        return ModelResponse(
            provider=self.config.provider,
            model=self.config.model,
            content="",
            error=message,
        )

    @staticmethod
    def _error_marker(message: str) -> str:
        return f"\n[ERROR: {message}]"
