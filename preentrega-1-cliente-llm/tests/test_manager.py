"""
Tests offline de AsyncLLMManager — no llaman a ninguna API real.

Se reemplaza el cliente interno por un FakeLLMClient que sigue un "guion"
(script) de respuestas y errores, para poder probar de forma determinística
la lógica de reintentos y el manejo de errores.

Correr con:  python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import AsyncIterator, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base_client import BaseLLMClient  # noqa: E402
from exceptions import LLMFatalError, LLMTransientError  # noqa: E402
from manager import AsyncLLMManager  # noqa: E402
from schemas import ChatMessage, ModelConfig, ModelResponse, Provider, Role  # noqa: E402


class FakeLLMClient(BaseLLMClient):
    """Cliente fake: en vez de llamar a un SDK real, sigue un guion fijo."""

    def __init__(self, config, generate_script=None, stream_script=None):
        super().__init__(config)
        self._generate_script = list(generate_script or [])
        self._stream_script = list(stream_script or [])
        self.generate_calls = 0
        self.stream_calls = 0

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        self.generate_calls += 1
        action = self._generate_script.pop(0)
        if isinstance(action, Exception):
            raise action
        return action

    async def stream(self, messages: List[ChatMessage]) -> AsyncIterator[str]:
        self.stream_calls += 1
        action = self._stream_script.pop(0)
        if isinstance(action, Exception):
            raise action
            yield ""  # pragma: no cover - inalcanzable; mantiene la función como generador
        for chunk in action:
            yield chunk

    async def aclose(self) -> None:
        pass


def _config(**overrides) -> ModelConfig:
    base = dict(provider=Provider.OPENAI, model="gpt-4o-mini", max_retries=2)
    base.update(overrides)
    return ModelConfig(**base)


def _manager_with_fake(config: ModelConfig, **kwargs):
    """Arma un AsyncLLMManager sin instanciar un cliente real (sin API key)."""
    manager = AsyncLLMManager.__new__(AsyncLLMManager)
    manager.config = config
    fake = FakeLLMClient(config, **kwargs)
    manager._client = fake
    return manager, fake


async def test_generate_camino_feliz():
    config = _config()
    ok = ModelResponse(provider=config.provider, model=config.model, content="42")
    manager, fake = _manager_with_fake(config, generate_script=[ok])

    resultado = await manager.generate([ChatMessage(role=Role.USER, content="hola")])

    assert resultado.ok
    assert resultado.content == "42"
    assert fake.generate_calls == 1


async def test_generate_recupera_de_error_transitorio():
    config = _config(max_retries=2)
    ok = ModelResponse(provider=config.provider, model=config.model, content="ok")
    manager, fake = _manager_with_fake(
        config, generate_script=[LLMTransientError("rate limit"), ok]
    )

    resultado = await manager.generate([ChatMessage(role=Role.USER, content="hola")])

    assert resultado.ok
    assert resultado.content == "ok"
    assert fake.generate_calls == 2  # 1 intento fallido + 1 reintento exitoso


async def test_generate_no_reintenta_ante_error_fatal():
    config = _config(max_retries=3)
    manager, fake = _manager_with_fake(
        config, generate_script=[LLMFatalError("API key inválida")]
    )

    resultado = await manager.generate([ChatMessage(role=Role.USER, content="hola")])

    assert not resultado.ok
    assert "API key inválida" in resultado.error
    assert fake.generate_calls == 1  # no debe reintentar un error fatal


async def test_generate_agota_reintentos_y_devuelve_error_estructurado():
    config = _config(max_retries=1)
    manager, fake = _manager_with_fake(
        config,
        generate_script=[LLMTransientError("timeout"), LLMTransientError("timeout")],
    )

    resultado = await manager.generate([ChatMessage(role=Role.USER, content="hola")])

    assert not resultado.ok
    assert "timeout" in resultado.error
    assert fake.generate_calls == 2  # intento inicial + 1 reintento (max_retries=1)


async def test_stream_entrega_fragmentos_en_orden():
    config = _config()
    manager, fake = _manager_with_fake(config, stream_script=[["Hola", " ", "mundo"]])

    fragmentos = [
        f async for f in manager.stream([ChatMessage(role=Role.USER, content="hola")])
    ]

    assert "".join(fragmentos) == "Hola mundo"


async def test_stream_nunca_propaga_excepcion_ante_error_fatal():
    config = _config(max_retries=0)
    manager, fake = _manager_with_fake(
        config, stream_script=[LLMFatalError("modelo inexistente")]
    )

    # Si esto no lanza, ya probamos el requisito: el loop principal no explota.
    fragmentos = [
        f async for f in manager.stream([ChatMessage(role=Role.USER, content="hola")])
    ]

    assert any("ERROR" in f for f in fragmentos)


async def test_stream_recupera_de_error_transitorio_antes_del_primer_chunk():
    config = _config(max_retries=1)
    manager, fake = _manager_with_fake(
        config,
        stream_script=[LLMTransientError("network blip"), ["todo", " ", "bien"]],
    )

    fragmentos = [
        f async for f in manager.stream([ChatMessage(role=Role.USER, content="hola")])
    ]

    assert "".join(fragmentos) == "todo bien"
    assert fake.stream_calls == 2
