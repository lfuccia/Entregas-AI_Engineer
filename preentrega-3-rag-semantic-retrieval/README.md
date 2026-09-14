# Sistema de Recuperación Semántica Local (RAG)

Sistema RAG end-to-end: indexa documentos técnicos en una base vectorial
local (ChromaDB), recupera los fragmentos más relevantes para una consulta
y genera una respuesta **fundamentada exclusivamente en esos fragmentos**
(si la respuesta no está en los documentos, el sistema lo dice
explícitamente en vez de inventar).

## Componentes

| Archivo | Qué hace |
|---|---|
| `data/*.md` | Dataset de ejemplo: 4 documentos técnicos sobre arquitectura de software y APIs |
| `embeddings.py` | Punto único de configuración del modelo de embeddings (mismo modelo para indexar y consultar) |
| `ingest.py` | Módulo de ingesta: carga `./data`, hace chunking (500 tokens / 50 de overlap) y persiste en ChromaDB (`./vectorstore`) |
| `rag_chain.py` | Retriever + prompt "filtro de veracidad" + cadena LCEL + `get_rag_response()` asíncrona |
| `schemas.py` | Modelo Pydantic `RespuestaRAG` (texto + fuentes citadas) |
| `test_rag.py` | Mini-script de prueba: una pregunta respondible y una "pregunta trampa" |
| `tests/` | Tests unitarios offline (sin LLM, sin descargar embeddings reales) |

## Cómo funciona el pipeline

```
./data/*.md
     │  (ingest.py)
     ▼
RecursiveCharacterTextSplitter.from_tiktoken_encoder
  chunk_size=500 tokens, chunk_overlap=50 tokens
     ▼
embeddings.get_embeddings()  (mismo modelo para indexar y consultar)
     ▼
ChromaDB persistente en ./vectorstore
     │
     │  (rag_chain.py, en tiempo de consulta)
     ▼
pregunta del usuario
     ▼
retriever.as_retriever(k=RAG_TOP_K)   # top_k entre 3 y 5
     ▼
contexto = fragmentos recuperados, etiquetados [FUENTE: archivo.md#fragmento_N]
     ▼
ChatPromptTemplate ("filtro de veracidad": solo responder con el CONTEXTO)
     ▼
LLM (Anthropic u OpenAI, configurable)
     ▼
PydanticOutputParser(RespuestaRAG)  (+ OutputFixingParser como red de seguridad)
     ▼
RespuestaRAG(respuesta=..., fuentes=[...])
```

### Decisiones clave (y los errores que evitan)

- **Embeddings no coincidentes:** `ingest.py` y
  `rag_chain.py` NUNCA instancian su propio modelo de embeddings; ambos
  importan `get_embeddings()` desde `embeddings.py`. Es imposible que
  indexen con un modelo y consulten con otro.
- **Contexto infinito / "Lost in the Middle":** el retriever trae `RAG_TOP_K`
  fragmentos (default 4, configurable entre 3 y 5 vía `.env`), nunca la
  base entera.
- **Falta de persistencia:** `ingest.py` chequea si `./vectorstore` ya
  tiene documentos indexados (`_vectorstore_ya_poblado`) y **no vuelve a
  indexar** salvo que se corra con `--force` o `ejecutar_ingesta(force=True)`.
- **Chunking basado en tokens, no en caracteres:** se usa
  `RecursiveCharacterTextSplitter.from_tiktoken_encoder` para que
  `chunk_size=500` y `chunk_overlap=50` sean tokens reales, tal como pide
  el enunciado, independientemente del proveedor de embeddings/LLM elegido.
- **Trazabilidad de fuentes:** cada fragmento indexado guarda
  `metadata["fuente_id"] = "archivo.md#fragmento_N"`. El prompt le muestra
  esos identificadores al LLM y le pide que solo cite los que realmente usó,
  y `RespuestaRAG.fuentes` los persiste en el objeto de salida validado.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

Por defecto (`EMBEDDINGS_PROVIDER=huggingface`) los embeddings corren
localmente con `sentence-transformers/all-MiniLM-L6-v2` — no hace falta
ninguna API key para esta parte, aunque la primera vez descarga el modelo
(~90 MB) desde Hugging Face.

Para la **generación** de la respuesta sí hace falta un LLM de verdad.
Editá `.env` y completá según el proveedor que uses:

```dotenv
LLM_PROVIDER=anthropic          # o "openai"
ANTHROPIC_API_KEY=sk-ant-...
# o bien:
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-...
```

## Uso

### 1. Indexar los documentos

```bash
python ingest.py
```

La primera vez indexa los 4 archivos de `./data` en ChromaDB. Si lo corrés
de nuevo, detecta que `./vectorstore` ya existe y **no vuelve a indexar**
(salvo `python ingest.py --force`).

### 2. Consultar

```python
import asyncio
from rag_chain import get_rag_response

async def main():
    resultado = await get_rag_response(
        "¿Qué algoritmo se recomienda para el rate limiting de una API?"
    )
    print(resultado.respuesta)
    print(resultado.fuentes)

asyncio.run(main())
```

### 3. Mini-script de prueba (pregunta respondible + pregunta trampa)

```bash
python test_rag.py
```

Corre dos consultas:

1. **Pregunta respondible** — "¿Qué algoritmo se recomienda para el rate
   limiting de una API y por qué es mejor que una ventana fija?" (la
   respuesta está en `03_seguridad_y_autenticacion.md`).
2. **Pregunta trampa** — "¿Qué estrategia de sharding recomienda el
   documento para escalar horizontalmente la base de datos?" (ninguno de
   los 4 documentos menciona sharding: el sistema debe responder que no
   tiene esa información, no inventarla).

### Ejemplo de salida esperada

**Pregunta respondible:**
```json
{
  "respuesta": "Se recomienda el algoritmo de 'token bucket': cada cliente tiene un balde con capacidad fija que se recarga a tasa constante, y a diferencia de una ventana fija permite ráfagas cortas de tráfico legítimo sin penalizar al cliente el resto de la ventana.",
  "fuentes": ["03_seguridad_y_autenticacion.md#fragmento_1"]
}
```

**Pregunta trampa:**
```json
{
  "respuesta": "No tengo acceso a esa información en los documentos disponibles: el contexto no menciona ninguna estrategia de sharding o particionamiento horizontal de la base de datos.",
  "fuentes": []
}
```

### Tests unitarios (offline, sin LLM)

```bash
pytest -v
```

Verifican el esquema Pydantic, la carga/chunking de documentos (con una
carpeta de datos temporal, sin tocar `./data` ni descargar el modelo de
embeddings real) y el formateo del contexto con las etiquetas `[FUENTE: ...]`.

## Variables de entorno (`.env`)

| Variable | Descripción | Default |
|---|---|---|
| `EMBEDDINGS_PROVIDER` | `huggingface` (local, gratis) u `openai` | `huggingface` |
| `HUGGINGFACE_EMBEDDING_MODEL` | Modelo de sentence-transformers | `sentence-transformers/all-MiniLM-L6-v2` |
| `LLM_PROVIDER` | `anthropic` u `openai` (modelo de generación) | `anthropic` |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | Credenciales/modelo de Anthropic | — |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | Credenciales/modelo de OpenAI | — |
| `DATA_DIR` | Carpeta con los documentos fuente | `./data` |
| `CHROMA_PERSIST_DIR` | Carpeta de persistencia de ChromaDB | `./vectorstore` |
| `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` | Tamaño y overlap del chunking (en tokens) | `500` / `50` |
| `RAG_TOP_K` | Cantidad de fragmentos recuperados por consulta | `4` |

## Checklist de entrega

- [x] Repositorio con script de ingesta (`ingest.py`: chunking + ChromaDB).
- [x] Script con la cadena RAG asíncrona (`rag_chain.py`: `get_rag_response()`).
- [x] Dataset de ejemplo (`data/*.md`, 4 archivos sobre arquitectura de software).
- [x] `README.md` con instrucciones de ejecución.
- [x] Chunking mínimo 500 tokens / 50 de overlap (`RecursiveCharacterTextSplitter.from_tiktoken_encoder`).
- [x] Mismo modelo de embeddings para indexar y consultar (`embeddings.py` como fuente única).
- [x] `top_k` entre 3 y 5 (`RAG_TOP_K=4` por defecto).
- [x] Verificación de persistencia antes de re-indexar (`_vectorstore_ya_poblado`).
- [x] Prompt "filtro de veracidad" (dice "no lo sé" si la respuesta no está en el contexto).
- [x] Salida parseada con `PydanticOutputParser` (`RespuestaRAG`: texto + referencias).
- [x] Dos pruebas en `test_rag.py`: pregunta respondible + pregunta trampa.
- [x] Sin API keys hardcodeadas: todo vía `.env` (`.env.example` incluido, `.env` en `.gitignore`).
