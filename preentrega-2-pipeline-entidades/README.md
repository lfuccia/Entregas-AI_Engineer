# Pipeline de Extracción de Entidades Técnicas

Pre-entrega 2: pipeline que recibe un párrafo de texto sin procesar (descripción
de arquitectura, log de error, ticket de soporte, etc.) y devuelve un objeto
validado con Pydantic.

## Estructura

| Archivo | Contenido |
|---|---|
| `schema.py` | El **contrato** de salida: `EntidadesTecnicas` (Pydantic), con `tecnologias`, `nivel_de_criticidad` (enum) y `resumen_tecnico`, más validadores custom. |
| `pipeline.py` | Prompt template modular + las dos cadenas LCEL (`build_structured_chain` y `build_parser_chain`) + `.with_retry()` + `get_model()`. |
| `test_pipeline.py` | **Prueba de estrés 100% offline** (sin API key) con un LLM fake, que demuestra la recuperación ante JSON mal formado/incompleto y el rechazo controlado ante un caso adversarial. |
| `demo_real.py` | Demo end-to-end contra un modelo real (Anthropic u OpenAI). |
| `requirements.txt` | Dependencias. |

## Los 4 componentes pedidos

1. **Esquema Pydantic** (`schema.py`): `EntidadesTecnicas` con `tecnologias: List[str]`
   (no puede estar vacía — validador custom), `nivel_de_criticidad: NivelCriticidad`
   (enum `baja` / `media` / `alta`) y `resumen_tecnico: str` (no vacío, mínimo 10
   caracteres).

2. **Prompt Template modular** (`pipeline.py::PROMPT_TEMPLATE`): acepta dos
   variables, `texto_entrada` y `format_instructions`. Esto permite reusar el
   mismo prompt tanto con `PydanticOutputParser` (que necesita el JSON schema
   como texto) como con `with_structured_output` (que no lo necesita, porque
   el schema viaja como tool/function call) — en ese caso se "parcializa" con
   una nota corta en vez del schema completo.

3. **Cadena LCEL** (`pipeline.py`):
   - `build_structured_chain(model)` — **la preferida**: `prompt | model.with_structured_output(EntidadesTecnicas)`.
     Usa function/tool calling nativo de Anthropic/OpenAI, que ya garantiza
     JSON válido contra el schema.
   - `build_parser_chain(model)` — alternativa con `PydanticOutputParser`
     (`prompt | model | parser`), para proveedores sin function calling o
     para observar explícitamente la recuperación ante texto mal formado.

4. **Lógica de resiliencia**: ambas cadenas se envuelven con
   `.with_retry(retry_if_exception_type=(ValidationError, OutputParserException, ValueError), stop_after_attempt=3, wait_exponential_jitter=True)`.
   Si el LLM devuelve JSON mal formado, JSON incompleto, o un valor que viola
   las reglas de negocio del schema (ej. lista de tecnologías vacía), la
   excepción se captura y se reintenta automáticamente hasta 3 veces antes de
   propagarse.

## Cómo correrlo

```bash
pip install -r requirements.txt

# 1) Prueba de estrés offline (no requiere API key):
python -m pytest test_pipeline.py -v

# 2) Demo con un modelo real:
export ANTHROPIC_API_KEY="sk-ant-..."   # o OPENAI_API_KEY
python demo_real.py
```

## Prueba de estrés — qué se verificó

`test_pipeline.py` reemplaza el LLM real por `FakeListChatModel` (respuestas
"enlatadas" y deterministas) para poder probar la lógica de resiliencia sin
depender de la no-determinismo de un modelo real ni de tener una API key:

- **Camino feliz**: JSON válido en el primer intento → no hace falta reintentar.
- **JSON mal formado** (sintaxis inválida, corte abrupto) en el primer intento,
  corregido en el segundo → `.with_retry()` recupera la cadena.
- **JSON incompleto** (faltan campos requeridos del schema) en el primer
  intento, corregido en el segundo → misma recuperación.
- **Caso adversarial**: el modelo insiste, en *todos* los intentos, en devolver
  una lista de tecnologías vacía (JSON sintácticamente válido, pero que viola
  la regla de negocio del validador). Acá el pipeline **no debe** inventar una
  respuesta: agotados los 3 intentos, propaga la excepción
  (`OutputParserException`) en lugar de devolver un objeto inválido.
- **Límite de reintentos**: si el error persiste más veces que `max_intentos`,
  la cadena también falla (no reintenta infinito).

Este último punto es a propósito: la resiliencia debe cubrir fallas
*transitorias* (un JSON mal formado por una mala racha del modelo), no
enmascarar un texto de entrada genuinamente inservible. Por eso el validador
de `tecnologias` sigue rechazando la respuesta aunque el "reintento" ya se
haya ejecutado — es la señal correcta de que hace falta pedirle al usuario un
texto con más contexto, no de que el sistema esté roto.

## Decisiones de diseño

- **`with_structured_output` como camino preferido**: al apoyarse en tool
  calling nativo, evita casi por completo el "JSON mal formado" (el proveedor
  ya fuerza la forma del objeto), dejando la resiliencia enfocada en errores
  de red y en las reglas de negocio del validador.
- **`build_parser_chain` como alternativa explícita**: se mantiene porque (a)
  es el mecanismo que pide el enunciado (`PydanticOutputParser`), (b) permite
  testear la recuperación ante JSON mal formado con un LLM fake sin necesitar
  tool calling, y (c) sirve para modelos/proveedores que no soporten function
  calling.
- **`get_model()` agnóstico de proveedor**: prioriza Anthropic y cae a OpenAI
  según la API key disponible, para no atar el pipeline a un solo proveedor.
