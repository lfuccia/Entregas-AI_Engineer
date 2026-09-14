# Pre-entrega 5 — Agente de razonamiento cíclico con memoria persistente

Agente ReAct construido con **LangGraph**: decide solo cuándo llamar a una
herramienta (sin ningún `if/else` manual), puede pedir aclaraciones cuando
una herramienta devuelve información ambigua o incompleta, y recuerda el
historial de la conversación entre llamadas gracias a un checkpointer
SQLite indexado por `thread_id`.

## Qué hace

Simula un asistente interno de una empresa que responde preguntas sobre
**clientes y sus pedidos**, usando dos herramientas propias (no hay ninguna
base de datos real: son diccionarios en memoria, a propósito, para que el
proyecto se pueda correr sin infraestructura extra):

- `buscar_cliente(nombre_o_id)`: resuelve el ID de un cliente a partir de
  su nombre (o confirma el ID si ya te lo dieron). Si el nombre es
  ambiguo (por ejemplo "Gómez", que matchea a dos personas), devuelve los
  candidatos en vez de adivinar.
- `buscar_pedidos(cliente_id)`: devuelve la cantidad de pedidos, el total
  facturado y el último pedido de un cliente.

El LLM decide solo, mirando el docstring de cada herramienta, cuál usar y
cuándo — incluyendo encadenar las dos (nombre → ID → pedidos) cuando hace
falta, o pedir una aclaración al usuario en vez de inventar un dato cuando
una herramienta devuelve algo ambiguo.

## Arquitectura

```
                ┌───────────────┐
   entrada ───▶ │    agente     │◀──────────────┐
                │ (LLM + tools) │                │
                └───────┬───────┘                │
                        │ tools_condition         │
              ¿tool_calls? sí          no         │
                        │              │          │
                        ▼              ▼          │
                ┌───────────────┐   END            │
                │  herramientas │───────────────────┘
                │  (ToolNode)   │
                └───────────────┘
```

- **Estado**: `MessagesState` (built-in de LangGraph) — la lista de
  mensajes se acumula sola vía el reducer `add_messages` que ya trae
  incorporado.
- **Nodo "agente"** (`graph.py`): llama al LLM con `.bind_tools([...])`.
  El propio modelo decide si responde directo o pide ejecutar una
  herramienta.
- **Nodo "herramientas"**: `ToolNode` (prebuilt de LangGraph) ejecuta la
  herramienta pedida y agrega el resultado como `ToolMessage`.
- **Arista condicional**: `tools_condition` (prebuilt) manda a
  "herramientas" si el LLM pidió una tool, o corta a `END` si ya
  respondió. Este es el ciclo: `agente → herramientas → agente → ... → END`.
- **Persistencia**: `AsyncSqliteSaver` (la variante asíncrona de
  `SqliteSaver`, del mismo paquete `langgraph-checkpoint-sqlite` —
  necesaria porque todo el proyecto corre con `asyncio`/`ainvoke`) guarda
  el estado en un archivo `.sqlite` real, indexado por `thread_id`. Usar
  el mismo `thread_id` en llamadas sucesivas hace que el agente "recuerde"
  la conversación anterior; usar uno distinto arranca de cero.
- **Límite de pasos**: cada invocación pasa `recursion_limit` (por
  defecto 10, configurable por `.env`) para evitar un bucle infinito
  costoso si algo sale mal.

## Requisitos

- Python 3.12+
- Una API key gratuita de [Groq](https://console.groq.com/keys) (si ya
  hiciste la pre-entrega 4, es la misma que usaste ahí — podés reusarla
  tal cual).

## Instalación

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Editá .env y completá GROQ_API_KEY con tu key real.
```

## Cómo probarlo

### 1. Charla interactiva

```bash
python main.py
```

Te va a pedir un `thread_id` (podés dejarlo vacío para usar `"default"`).
Probá, por ejemplo:

```
Vos: ¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?
Agente: El cliente 102 tuvo 3 pedidos por un total de $14.500.

Vos: ¿Y cuál fue el último pedido?
Agente: El último pedido del cliente 102 fue el P-2044, del 2026-08-30, por $5.200.
```

Si volvés a correr `python main.py` y usás el **mismo** `thread_id`, el
agente sigue recordando esa conversación (queda guardada en
`checkpoints.sqlite`). Con un `thread_id` distinto, arranca de cero.

Para ver el razonamiento multi-paso (una sola pregunta que dispara dos
herramientas encadenadas):

```
Vos: ¿Cuántos pedidos tuvo Martina Gómez y por cuánto total?
```

Y para ver el ciclo de retorno (el agente pide aclaración en vez de
inventar):

```
Vos: ¿Cuántos pedidos tuvo el cliente Gómez?
Agente: Hay más de un cliente con apellido Gómez: Martina Gómez (103) y Diego Gómez (104). ¿A cuál te referís?
Vos: Al 104, Diego Gómez.
Agente: Diego Gómez (cliente 104) no tiene pedidos registrados todavía.
```

### 2. Tests offline (sin API key, sin red)

```bash
pytest
```

Corre 14 tests: lógica de las herramientas, selección de LLM por
proveedor, y el ruteo completo del grafo (agente ↔ herramientas ↔
persistencia) usando un LLM falso y determinístico — no gastan ninguna
llamada real.

### 3. Traza de ejemplo ya incluida en el repo

`trace_ejemplo.json` (en la raíz del repo) ya viene generado y versionado:
es la salida real de correr el grafo completo (`StateGraph` + `ToolNode` +
`AsyncSqliteSaver`, todo real) con un LLM **determinístico** de prueba
(`demo_llm.py`) en vez de uno real, para que el repo tenga evidencia
reproducible sin depender de ninguna API key. Cubre los tres escenarios
pedidos:

- **Escenario A** — memoria persistente: dos preguntas en el mismo
  `thread_id`; la segunda se responde con datos ya obtenidos en la
  primera, sin volver a llamar a la herramienta.
- **Escenario B** — razonamiento multi-paso: una sola pregunta que
  encadena `buscar_cliente` → `buscar_pedidos` (dos llamadas a
  herramientas distintas) antes de concluir.
- **Escenario C** — ciclo de retorno: `buscar_cliente` devuelve varios
  candidatos ambiguos y el agente pide aclaración en vez de inventar; en
  el turno siguiente (mismo `thread_id`) retoma con la aclaración.

Para regenerarlo vos mismo:

```bash
python demo_offline.py
```

### 4. Traza con el LLM real (evidencia "de verdad")

```bash
python run_demo.py
```

Corre los mismos tres escenarios pero con el LLM real configurado en tu
`.env` (Groq por defecto), y escribe `trace_real.json` (no se versiona en
git — cada uno genera la suya con su propia key). A diferencia de
`trace_ejemplo.json`, acá el razonamiento es genuino: el modelo puede
elegir un camino ligeramente distinto en cada corrida.

2026-09-14 12:25:11,618 | INFO     | [A - memoria] Usuario: ¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?
2026-09-14 12:25:12,511 | INFO     | HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-14 12:25:12,941 | INFO     | HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-14 12:25:12,945 | INFO     | [A - memoria] Agente: El cliente 102 tuvo **3 pedidos** y el total acumulado es **$14 500**.
2026-09-14 12:25:12,945 | INFO     | [A - memoria] Usuario: ¿Y cuál fue el último pedido?
2026-09-14 12:25:13,458 | INFO     | HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-14 12:25:13,462 | INFO     | [A - memoria] Agente: El último pedido del cliente 102 es el **P‑2044**, realizado el **30 de agosto de 2026** con un monto de **$5 200**.
2026-09-14 12:25:13,462 | INFO     | [B - multi-paso] Usuario: ¿Cuántos pedidos tuvo Martina Gómez y por cuánto total?
2026-09-14 12:25:13,987 | INFO     | HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-14 12:25:14,394 | INFO     | HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-14 12:25:14,838 | INFO     | HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-14 12:25:14,843 | INFO     | [B - multi-paso] Agente: Martina Gómez (ID 103) tuvo **7 pedidos** con un total de **$58 900**.
2026-09-14 12:25:14,843 | INFO     | [C - aclaración] Usuario: ¿Cuántos pedidos tuvo el cliente Gómez?
2026-09-14 12:25:15,495 | INFO     | HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-14 12:25:16,345 | INFO     | HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-14 12:25:16,350 | INFO     | [C - aclaración] Agente: Veo que hay varios clientes con el apellido **Gómez**. ¿Te referís a:

- **Martina Gómez** (ID 103)  
- **Diego Gómez** (ID 104)  

Por favor, indicá cuál de los dos querés consultar.
2026-09-14 12:25:16,351 | INFO     | [C - aclaración] Usuario: Al 104, Diego Gómez.
2026-09-14 12:25:16,423 | INFO     | HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-09-14 12:25:16,424 | INFO     | Retrying request to /openai/v1/chat/completions in 1.000000 seconds
2026-09-14 12:25:17,957 | INFO     | HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-14 12:25:18,038 | INFO     | HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-09-14 12:25:18,039 | INFO     | Retrying request to /openai/v1/chat/completions in 9.000000 seconds
2026-09-14 12:25:27,787 | INFO     | HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-09-14 12:25:27,793 | INFO     | [C - aclaración] Agente: El cliente **Diego Gómez** (ID 104) no tiene pedidos registrados todavía (0 pedidos, total $0).
2026-09-14 12:25:27,799 | INFO     | Traza escrita en 'trace_real.json'.

$ python run_demo.py
...
-->

## Estructura del repo

```
config.py            # configuración desde variables de entorno (.env)
llm.py                # selector de LLM (Groq / OpenAI / Anthropic)
tools.py              # @tool: buscar_cliente, buscar_pedidos
graph.py              # StateGraph: nodo agente + ToolNode + tools_condition
demo_llm.py           # LLM determinístico de prueba (solo para demo_offline.py)
demo_offline.py       # genera trace_ejemplo.json (sin API key)
run_demo.py           # genera trace_real.json (con tu API key real)
main.py               # REPL interactivo
trace_ejemplo.json    # evidencia ya incluida en el repo
tests/                # tests offline (pytest)
```

## Checklist de la consigna

- [x] Repo con `StateGraph` que hereda de `MessagesState`, nodo de modelo
      + nodo de herramientas + arista condicional `tools_condition`.
- [x] Al menos 1 herramienta con `@tool` y docstring descriptivo (hay 2:
      `buscar_cliente`, `buscar_pedidos`).
- [x] Persistencia con `SqliteSaver` (su variante async, `AsyncSqliteSaver`,
      requerida por correr todo con `asyncio`) + `thread_id`.
- [x] Prueba con razonamiento multi-paso (la herramienta se invoca ≥2
      veces en una sola pregunta) y `recursion_limit` definido en cada
      invocación.
- [x] Ejemplo de traza de ejecución incluido (`trace_ejemplo.json`).
- [x] Python 3.12, type hints en todo el código, `asyncio` de punta a
      punta (`ainvoke`, nodos `async def`).
- [x] `.env` fuera de git (`.gitignore`); `.env.example` como plantilla.

### Extra (no pedido explícitamente por el enunciado)

- Selector de LLM por variable de entorno (`LLM_PROVIDER=groq|openai|anthropic`)
  para poder correr todo con Groq gratis y probarlo de verdad, sin tarjeta.
- `demo_offline.py`: evidencia reproducible sin API key, además de
  `run_demo.py` con el LLM real.
- Tests de ruteo del grafo con inyección de dependencias (LLM falso), sin
  gastar llamadas reales — mismo patrón que las pre-entregas anteriores.
