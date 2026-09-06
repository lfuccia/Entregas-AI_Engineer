"""
Script de validación.

Carga la configuración desde `.env`, instancia el `AsyncLLMManager` con el
proveedor elegido, y le hace una pregunta corta tanto en modo normal como
en modo streaming.

Ejecutar:
    python main.py
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from dotenv import load_dotenv

from manager import AsyncLLMManager
from schemas import ChatMessage, ModelConfig, Provider, Role

PREGUNTA = "¿Qué es la entropía?"

_DEFAULT_MODELS = {
    Provider.OPENAI: "gpt-4o-mini",
    Provider.ANTHROPIC: "claude-sonnet-4-5",
}


def _build_config() -> ModelConfig:
    provider = Provider(os.getenv("LLM_PROVIDER", "openai").strip().lower())
    return ModelConfig(
        provider=provider,
        model=os.getenv("LLM_MODEL", _DEFAULT_MODELS[provider]),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "512")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
    )


def _api_key_for(provider: Provider) -> Optional[str]:
    env_var = "OPENAI_API_KEY" if provider == Provider.OPENAI else "ANTHROPIC_API_KEY"
    return os.getenv(env_var)


async def main() -> None:
    load_dotenv()
    config = _build_config()
    api_key = _api_key_for(config.provider)

    if not api_key:
        env_var = "OPENAI_API_KEY" if config.provider == Provider.OPENAI else "ANTHROPIC_API_KEY"
        print(
            f"No se encontró {env_var} en el entorno.\n"
            f"Copiá .env.example a .env y completá la clave de '{config.provider.value}'."
        )
        return

    manager = AsyncLLMManager(config, api_key=api_key)
    mensajes = [ChatMessage(role=Role.USER, content=PREGUNTA)]

    try:
        print(f"Proveedor: {config.provider.value} | Modelo: {config.model}")
        print(f"Pregunta: {PREGUNTA}\n")

        print("=" * 70)
        print("MODO NORMAL (generate)")
        print("=" * 70)
        respuesta = await manager.generate(mensajes)
        if not respuesta.ok:
            print(f"Error: {respuesta.error}")
        else:
            print(respuesta.content)
            print(f"\n[uso de tokens: {respuesta.usage}]")

        print()
        print("=" * 70)
        print("MODO STREAMING (stream)")
        print("=" * 70)
        async for fragmento in manager.stream(mensajes):
            print(fragmento, end="", flush=True)
        print()
    finally:
        await manager.aclose()


if __name__ == "__main__":
    asyncio.run(main())
