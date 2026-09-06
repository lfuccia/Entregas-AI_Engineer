"""
Cliente asíncrono para Anthropic (Claude). Implementa BaseLLMClient usando
AsyncAnthropic.
"""

from __future__ import annotations

from typing import AsyncIterator, List, Optional, Tuple

import anthropic
from anthropic import AsyncAnthropic

from base_client import BaseLLMClient
from exceptions import LLMFatalError, LLMStreamInterrupted, LLMTransientError
from schemas import ChatMessage, ModelConfig, ModelResponse, Provider, Role

# Errores de red / cuota / disponibilidad transitoria del servicio: reintentables.
_TRANSIENT_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
    anthropic.OverloadedError,
)


class AnthropicClient(BaseLLMClient):
    """Envuelve `AsyncAnthropic` detrás de la interfaz `BaseLLMClient`."""

    def __init__(self, config: ModelConfig, api_key: Optional[str] = None) -> None:
        super().__init__(config, api_key)
        self._client = AsyncAnthropic(
            api_key=api_key,
            timeout=config.timeout,
            max_retries=0,  # los reintentos los maneja AsyncLLMManager
        )

    @staticmethod
    def _split_system(messages: List[ChatMessage]) -> Tuple[Optional[str], list[dict]]:
        """Anthropic recibe el system prompt aparte, no como un mensaje más."""
        system_parts = [m.content for m in messages if m.role == Role.SYSTEM]
        rest = [
            {"role": m.role.value, "content": m.content}
            for m in messages
            if m.role != Role.SYSTEM
        ]
        system = "\n".join(system_parts) if system_parts else None
        return system, rest

    def _build_kwargs(self, messages: List[ChatMessage]) -> dict:
        system, rest = self._split_system(messages)
        kwargs: dict = dict(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=rest,
        )
        if system:
            kwargs["system"] = system
        if self.config.top_p is not None:
            kwargs["top_p"] = self.config.top_p
        return kwargs

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        try:
            response = await self._client.messages.create(**self._build_kwargs(messages))
        except _TRANSIENT_EXCEPTIONS as exc:
            raise LLMTransientError(str(exc)) from exc
        except anthropic.AnthropicError as exc:
            raise LLMFatalError(str(exc)) from exc

        content = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        usage = response.usage
        return ModelResponse(
            provider=Provider.ANTHROPIC,
            model=response.model,
            content=content,
            finish_reason=response.stop_reason,
            usage=(
                {
                    "prompt_tokens": usage.input_tokens,
                    "completion_tokens": usage.output_tokens,
                    "total_tokens": usage.input_tokens + usage.output_tokens,
                }
                if usage
                else None
            ),
        )

    async def stream(self, messages: List[ChatMessage]) -> AsyncIterator[str]:
        emitted_any = False
        try:
            async with self._client.messages.stream(**self._build_kwargs(messages)) as stream:
                async for text in stream.text_stream:
                    emitted_any = True
                    yield text
        except _TRANSIENT_EXCEPTIONS as exc:
            raise (LLMStreamInterrupted if emitted_any else LLMTransientError)(str(exc)) from exc
        except anthropic.AnthropicError as exc:
            raise (LLMStreamInterrupted if emitted_any else LLMFatalError)(str(exc)) from exc

    async def aclose(self) -> None:
        await self._client.close()
