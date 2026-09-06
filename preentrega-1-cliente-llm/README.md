# Unified Async LLM Client

Pre-entrega 1: cliente asíncrono y unificado para OpenAI y Anthropic, con
streaming, validación con Pydantic y manejo robusto de errores.

## Estructura del repositorio

| Archivo | Contenido |
|---|---|
| `schemas.py` | Pydantic: `ChatMessage`, `ModelConfig` (temperatura 0-2, `max_tokens`, etc.) y `ModelResponse`. |
| `exceptions.py` | Jerarquía propia de errores (`LLMTransientError`, `LLMFatalError`, `LLMStreamInterrupted`) — desacopla al manager de los SDKs de cada proveedor. |
| `base_client.py` | `BaseLLMClient`, la clase base abstracta (`generate` / `stream` / `aclose`). |
| `openai_client.py` | `OpenAIClient`, implementación con `AsyncOpenAI`. |
| `anthropic_client.py` | `AnthropicClient`, implementación con `AsyncAnthropic`. |
| `manager.py` | `AsyncLLMManager`: elige el proveedor y agrega reintentos + manejo de errores. |
| `main.py` | Script de validación: pregunta "¿Qué es la entropía?" en modo normal y en streaming. |
| `tests/test_manager.py` | Pruebas offline (sin API key) de la lógica de reintentos, con un cliente fake. |
| `.env.example` | Variables de entorno necesarias. |

## Requisitos

- Python **3.12**
- Una API key de OpenAI y/o de Anthropic

## Instalación

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt          # solo lo necesario para correr main.py
# o, para correr también los tests:
pip install -r requirements-dev.txt
```

## Variables de entorno

Copiá `.env.example` a `.env` y completá los valores:

```bash
cp .env.example .env
```

| Variable | Obligatoria | Descripción |
|---|---|---|
| `LLM_PROVIDER` | No (default `openai`) | `openai` o `anthropic`. |
| `LLM_MODEL` | No | Modelo a usar. Default: `gpt-4o-mini` (OpenAI) / `claude-sonnet-4-5` (Anthropic). |
| `LLM_TEMPERATURE` | No (default `0.7`) | Entre 0 y 2. |
| `LLM_MAX_TOKENS` | No (default `512`) | Tokens máximos de la respuesta. |
| `LLM_MAX_RETRIES` | No (default `2`) | Reintentos ante errores transitorios. |
| `OPENAI_API_KEY` | Sí, si `LLM_PROVIDER=openai` | Tu clave de OpenAI. |
| `ANTHROPIC_API_KEY` | Sí, si `LLM_PROVIDER=anthropic` | Tu clave de Anthropic. |

`.env` está en `.gitignore`: nunca se sube al repositorio.

## Cómo correrlo

```bash
# Prueba de estrés / lógica de reintentos, 100% offline (no requiere API key):
python -m pytest tests/ -v

# Script de validación contra un modelo real (requiere .env configurado):
python main.py
```

`main.py` imprime la respuesta a "¿Qué es la entropía?" primero en modo
normal (`generate`) y después en modo streaming (`stream`), token a token,
usando el mismo proveedor y modelo configurados en `.env`.

Para probar el otro proveedor sin tocar código, cambiá `LLM_PROVIDER` en
`.env` (y asegurate de tener la API key correspondiente) — esa es la
"intercambiabilidad" que pide el enunciado.

## Diseño

### 1. Intercambiabilidad

`BaseLLMClient` define la interfaz común (`generate`, `stream`, `aclose`).
`OpenAIClient` y `AnthropicClient` la implementan cada uno con su propio
SDK. `AsyncLLMManager` elige cuál instanciar según `config.provider`, así
que el resto del programa (`main.py`) nunca necesita saber qué proveedor
hay detrás.

### 2. Asincronía

Todo el pipeline usa `AsyncOpenAI` / `AsyncAnthropic` y `async/await` de
punta a punta — nunca se llama a la variante síncrona de ningún SDK dentro
de una función `async` (el error clásico que bloquea el event loop
completo mientras el LLM "piensa").

### 3. Streaming

`stream()` es un generador asíncrono (`async def ... yield`). Para OpenAI
itera el `AsyncStream[ChatCompletionChunk]` que devuelve
`chat.completions.create(..., stream=True)`; para Anthropic usa el context
manager `client.messages.stream(...)` y su `stream.text_stream`.

### 4. Validación con Pydantic

`ModelConfig` valida `temperature` (0-2), `max_tokens` (> 0) y demás
parámetros *antes* de que lleguen al SDK del proveedor. `ChatMessage`
obliga a que todo mensaje tenga `role` y `content`. `ModelResponse` es la
forma única en la que sale cualquier respuesta, sin importar el proveedor.

### Manejo de errores (sin fuga de excepciones)

Cada cliente traduce las excepciones específicas de su SDK a una jerarquía
propia (`exceptions.py`):

- `LLMTransientError` — red, timeout, rate limit, 5xx del proveedor → se
  reintenta con backoff exponencial (hasta `max_retries` veces).
- `LLMFatalError` — API key inválida, request mal formado, modelo
  inexistente → no tiene sentido reintentar, se corta al toque.
- `LLMStreamInterrupted` — el stream se cortó después de haber emitido ya
  contenido parcial (no se puede "deshacer" texto ya entregado).

`AsyncLLMManager.generate()` **nunca propaga una excepción**: ante
cualquier error, agotados los reintentos si correspondía, devuelve un
`ModelResponse` con `error` seteado (`response.ok is False`) en lugar de
romper el programa que lo llama.

`AsyncLLMManager.stream()` funciona igual en espíritu: si el error ocurre
antes de emitir el primer fragmento, reintenta como `generate()`; si ya se
emitió contenido, o el error es fatal, emite un último fragmento con el
error (`"\n[ERROR: ...]"`) y termina el generador — el `async for` de quien
consume el stream tampoco explota nunca.

## Pruebas offline (sin API key)

`tests/test_manager.py` reemplaza el cliente real por un `FakeLLMClient`
que sigue un guion de respuestas/errores predefinido, para poder probar de
forma determinística:

- Camino feliz (`generate` y `stream`).
- Recuperación ante un error transitorio (`LLMTransientError`) en el primer
  intento, éxito en el segundo.
- Un error fatal (`LLMFatalError`) **no** se reintenta.
- Agotados los reintentos configurados, se devuelve un error estructurado
  (no una excepción).
- El streaming nunca propaga una excepción, incluso ante un error fatal.

```bash
python -m pytest tests/ -v
```
