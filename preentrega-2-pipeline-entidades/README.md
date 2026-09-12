# Pipeline de Extracción de Entidades Técnicas

Pipeline RAG-adyacente que recibe un párrafo de texto sin procesar (una
descripción de arquitectura de software, un log de error, un ticket de
incidente, etc.) y devuelve un **objeto validado** con las tecnologías
mencionadas, el nivel de criticidad y un resumen técnico.

Construido con **LangChain (LCEL)** + **Pydantic**, con reintento
automático ante respuestas mal formadas o incompletas.

## Componentes

| Archivo | Qué contiene |
|---|---|
| `schemas.py` | Modelo Pydantic `EntidadesTecnicas` (el "contrato" de salida) |
| `chain.py` | Cliente LLM configurable, `ChatPromptTemplate`, cadena LCEL con `.with_structured_output()` y `.with_retry()` |
| `test_pipeline.py` | Mini-script de prueba asíncrono (requiere API key real) |
| `tests/` | Tests unitarios offline (no llaman a ningún LLM) |

## El esquema (`schemas.py`)

```python
class NivelCriticidad(str, Enum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"

class EntidadesTecnicas(BaseModel):
    tecnologias: list[str]              # nunca vacía, sin duplicados
    nivel_de_criticidad: NivelCriticidad # "baja" | "media" | "alta"
    resumen_tecnico: str                 # resumen breve, no vacío
```

Restricciones aplicadas con `field_validator`:
- `tecnologias` no puede quedar vacía (ni ser una lista de strings en
  blanco) y se deduplica (case-insensitive).
- `resumen_tecnico` no puede estar vacío y tiene un largo mínimo.

## La cadena LCEL (`chain.py`)

```python
modelo_estructurado = llm.with_structured_output(EntidadesTecnicas, include_raw=True)
chain = (PROMPT | modelo_estructurado | RunnableLambda(_validar_respuesta_cruda)).with_retry(
    retry_if_exception_type=(RespuestaIncompletaError, RespuestaMalFormadaError, ValidationError),
    stop_after_attempt=3,
)
```

- **Prompt modular**: `ChatPromptTemplate.from_messages([...])` con
  variables (`{texto_entrada}`, `{instrucciones_formato}`) — no hay
  f-strings de Python hardcodeadas armando el mensaje.
- **Structured output**: se usa `model.with_structured_output()` (método
  preferido, vía tool-calling nativo de OpenAI/Anthropic) en vez de
  `PydanticOutputParser` a mano. `PydanticOutputParser` se usa solo para
  generar las `instrucciones_formato` que refuerzan el prompt.
- **Resiliencia ante `finish_reason`**: se pasa `include_raw=True` para
  poder inspeccionar el `AIMessage` crudo. La función
  `_validar_respuesta_cruda()` chequea explícitamente
  `response_metadata["finish_reason"]` (o `stop_reason`) y, si el
  proveedor cortó la respuesta por tokens (`"length"` / `"max_tokens"`),
  la trata como **respuesta incompleta** y fuerza un reintento — en vez de
  intentar transformar un JSON truncado.
- **Resiliencia ante JSON mal formado**: si `parsing_error` viene seteado
  o `parsed` es `None`, también se levanta una excepción propia
  (`RespuestaMalFormadaError`) para disparar el reintento.
- **Reintento**: `.with_retry()` envuelve toda la cadena y reintenta
  automáticamente (backoff exponencial con jitter) ante
  `RespuestaIncompletaError`, `RespuestaMalFormadaError` o
  `pydantic.ValidationError`, hasta `PIPELINE_MAX_REINTENTOS` intentos
  (default: 3).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Editá .env: elegí LLM_PROVIDER=anthropic (u openai) y pegá tu API key.
```

## Uso

```python
import asyncio
from chain import process_text

async def main():
    resultado = await process_text(
        "Nuestro backend expone una API en FastAPI que atiende picos de "
        "5000 requests por segundo. Usamos Redis como caché y PostgreSQL "
        "como base de datos principal. Bajo carga, el pool de conexiones "
        "a PostgreSQL se agota y aparecen timeouts en producción."
    )
    print(resultado.model_dump_json(indent=2))

asyncio.run(main())
```

### Mini-script de prueba

Corre tres ejemplos (arquitectura clara, log de error crítico, y un texto
**ambiguo** como prueba de estrés) y muestra logs de validación/reintentos:

```bash
python test_pipeline.py
```

### Tests unitarios (offline, sin API key)

Verifican el esquema Pydantic y la lógica de resiliencia
(`finish_reason`, JSON mal formado) sin llamar a ningún LLM real:

```bash
pytest tests/ -v
```

## Ejemplo de salida esperada

Dado el texto de arquitectura del ejemplo de arriba, el pipeline devuelve
un objeto validado como este:

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL; cuello de botella en conexiones concurrentes."
}
```

## Prueba de estrés (texto ambiguo)

`test_pipeline.py` incluye un texto deliberadamente ambiguo ("el sistema
anduvo raro toda la mañana... nadie encontró nada raro en los
dashboards"). Dos comportamientos son válidos y ambos están contemplados:

- El modelo **se recupera** infiriendo una tecnología plausible del
  contexto (ej. `"HTTP"`, `"infraestructura web"`) para no violar la
  restricción de lista no vacía.
- Si, tras `PIPELINE_MAX_REINTENTOS` intentos, la respuesta sigue sin
  cumplir el esquema, el pipeline **propaga la excepción**
  (`pydantic.ValidationError` o el error de resiliencia correspondiente)
  en vez de devolver silenciosamente un objeto inválido o incompleto.

## Variables de entorno (`.env`)

| Variable | Descripción | Default |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` u `openai` | `anthropic` |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | Credenciales/modelo de Anthropic | — / `claude-sonnet-4-5-20250929` |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | Credenciales/modelo de OpenAI | — / `gpt-4o-mini` |
| `LLM_TEMPERATURE` | Temperatura del modelo | `0` |
| `PIPELINE_MAX_REINTENTOS` | Intentos máximos de `.with_retry()` | `3` |
| `LOG_LEVEL` | Nivel de logging | `INFO` |

## Checklist de entrega

- [x] Repositorio con `schemas.py` (modelo Pydantic) y `chain.py` (cadena LCEL).
- [x] Cadena compuesta con LCEL: `prompt | model.with_structured_output(Schema)`.
- [x] Lógica de reintento (`.with_retry()`) ante JSON mal formado o incompleto (incluye chequeo explícito de `finish_reason`).
- [x] Función asíncrona `process_text()` con `.ainvoke()` y logs de validación.
- [x] Mini-script de prueba (`test_pipeline.py`) que ejecuta varios ejemplos, incluida una prueba de estrés con texto ambiguo.
- [x] Tests unitarios offline adicionales (`tests/`) que verifican la lógica de resiliencia sin necesidad de API key.
