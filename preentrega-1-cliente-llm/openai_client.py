"""
Cliente asíncrono para OpenAI. Implementa BaseLLMClient usando AsyncOpenAI.
"""

from __future__ import annotations

from typing import AsyncIterator, List, Optional

import openai
from openai import AsyncOpenAI

from base_client import BaseLLMClient
from exceptions import LLMFatalError, LLMStreamInterrupted, LLMTransientError
from schemas import ChatMessage, ModelConfig, ModelResponse, Provider

# Errores de red / cuota / disponibilidad transitoria del servicio: reintentables.
_TRANSIENT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


class OpenAIClient(BaseLLMClient):
    """Envuelve `AsyncOpenAI` detrás de la interfaz `BaseLLMClient`."""

    def __init__(self, config: ModelConfig, api_key: Optional[str] = None) -> None:
        super().__init__(config, api_key)
        self._client = AsyncOpenAI(
            api_key=api_key,
            timeout=config.timeout,
            max_retries=0,  # los reintentos los maneja AsyncLLMManager
        )

    @staticmethod
    def _to_openai_messages(messages: List[ChatMessage]) -> list[dict]:
        return [{"role": m.role.value, "content": m.content} for m in messages]

    def _build_kwargs(self, messages: List[ChatMessage], *, stream: bool) -> dict:
        kwargs: dict = dict(
            model=self.config.model,
            messages=self._to_openai_messages(messages),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        if self.config.top_p is not None:
            kwargs["top_p"] = self.config.top_p
        if stream:
            kwargs["stream"] = True
        return kwargs

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        try:
            response = await self._client.chat.completions.create(
                **self._build_kwargs(messages, stream=False)
            )
        except _TRANSIENT_EXCEPTIONS as exc:
            raise LLMTransientError(str(exc)) from exc
        except openai.OpenAIError as exc:
            raise LLMFatalError(str(exc)) from exc

        choice = response.choices[0]
        usage = response.usage
        return ModelResponse(
            provider=Provider.OPENAI,
            model=response.model,
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=(
                {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
                if usage
                else None
            ),
        )

    async def stream(self, messages: List[ChatMessage]) -> AsyncIterator[str]:
        emitted_any = False
        try:
            response_stream = await self._client.chat.completions.create(
                **self._build_kwargs(messages, stream=True)
            )
        except _TRANSIENT_EXCEPTIONS as exc:
            raise LLMTransientError(str(exc)) from exc
        except openai.OpenAIError as exc:
            raise LLMFatalError(str(exc)) from exc

        try:
            async for chunk in response_stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    emitted_any = True
                    yield delta
        except _TRANSIENT_EXCEPTIONS as exc:
            raise (LLMStreamInterrupted if emitted_any else LLMTransientError)(str(exc)) from exc
        except openai.OpenAIError as exc:
            raise (LLMStreamInterrupted if emitted_any else LLMFatalError)(str(exc)) from exc

    async def aclose(self) -> None:
        await self._client.close()
