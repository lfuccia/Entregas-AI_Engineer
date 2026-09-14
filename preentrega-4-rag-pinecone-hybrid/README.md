# Sistema RAG escalable en la nube con Pinecone

Módulo de Recuperación Escalable: ingesta documentos técnicos en un índice
**Pinecone Serverless**, los recupera con un **Recuperador Híbrido**
(BM25 léxico + similitud vectorial, combinados con `EnsembleRetriever`) y
mide la calidad de esa recuperación con un **script de evaluación**
(Precision@5 / Recall@5) contra un Golden Set de preguntas.

**Todo el pipeline se puede probar de punta a punta sin pagar nada**:
los embeddings usan por defecto un modelo local de HuggingFace (gratis,
sin API key) y hay un paso opcional de generación de respuesta con Groq
(modelos Llama open-source, free tier real sin tarjeta de crédito). Lo
único que sigue requiriendo una cuenta (gratuita) es Pinecone.

## Componentes

| Archivo | Qué hace |
|---|---|
| `data/` | Dataset de ejemplo: documentación técnica de FastAPI/Pydantic en 3 formatos — `.md`, `.json` y `.pdf` |
| `config.py` | Única fuente de variables de entorno para todo el proyecto |
| `embeddings.py` | Modelo de embeddings — HuggingFace local **gratis** (default, 384 dims) u OpenAI (opcional, 1536 dims) — compartido por ingesta y consulta |
| `loaders.py` | Carga multi-formato (.md/.json/.pdf) con metadatos avanzados (fuente, página, categoría) |
| `pinecone_setup.py` | Verifica/crea el índice Pinecone Serverless; detecta mismatch de dimensiones |
| `ingest.py` | Chunking (500-800 tokens) + subida a Pinecone con namespace + metadata + texto original |
| `corpus_cache.py` | Persistencia local del corpus chunkeado (para BM25) |
| `rag_system.py` | Clase `RAGSystem`: `EnsembleRetriever` (BM25 + Pinecone) + `answer()`/`aanswer()` (generación opcional) |
| `generation.py` | **Bonus, no pedido por el enunciado**: genera una respuesta grounded con un LLM gratis (Groq/Llama) a partir de lo recuperado |
| `evaluate.py` | Precision@5 / Recall@5 contra `golden_set.json`, reporte en consola |
| `golden_set.json` | 5 preguntas con documento fuente conocido |
| `test_generacion.py` | Mini-script de evidencia end-to-end: recuperación + respuesta real generada |
| `tests/` | 32 tests unitarios offline (sin Pinecone, sin OpenAI, sin Groq) |

## Arquitectura

```
data/*.{md,json,pdf}
     │  (loaders.py: carga + metadata fuente/página/categoría)
     ▼
RecursiveCharacterTextSplitter.from_tiktoken_encoder
  chunk_size=650 tokens (rango 500-800), overlap=80 tokens
     │  (ingest.py: _fragmentar)
     ▼
     ├──> corpus_cache.jsonl  (copia local: texto + metadata, para BM25)
     │
     └──> embeddings.py
              │  default: HuggingFace local (all-MiniLM-L6-v2, 384 dims, GRATIS)
              │  alternativa: OpenAI (text-embedding-3-small, 1536 dims, pago)
              ▼
          Pinecone Serverless (namespace="documentacion-tecnica")
          vector + metadata {text, source, page, categoria, fuente_id}

                    ── tiempo de consulta ──

pregunta del usuario
     │
     ├──> BM25Retriever (léxico, sobre corpus_cache.jsonl)
     │
     └──> PineconeVectorStore.as_retriever (semántico, k=5)
              │
              ▼
     EnsembleRetriever(weights=[0.5, 0.5])   <- RAGSystem.query()
              │
              ▼
        top-5 Document combinados
              │
              ├──> evaluate.py: Precision@5 / Recall@5 contra golden_set.json
              │
              └──> (BONUS) generation.py + Groq/Llama  <- RAGSystem.answer()
                        │
                        ▼
                   respuesta grounded en texto + fuentes citadas
```

### Decisiones clave (y los errores que evitan)

- **Embeddings gratis por default:** `EMBEDDINGS_PROVIDER=huggingface`
  corre `sentence-transformers/all-MiniLM-L6-v2` localmente (384 dims),
  sin API key ni costo — así se puede correr `ingest.py`/`evaluate.py` de
  punta a punta sin depender de una cuenta de pago. `EMBEDDINGS_PROVIDER=openai`
  queda disponible como alternativa (1536 dims, mejor calidad semántica).
- **Mismatch de dimensiones:** `pinecone_setup.py` valida explícitamente
  que la dimensión del índice existente coincida con
  `EMBEDDING_DIMENSION` (384 u 1536 según el proveedor elegido) y corta con
  un error claro (`DimensionMismatchError`) si no coinciden, en vez de
  dejar que cada upsert falle uno por uno.
- **Namespace siempre explícito:** todo el índice usa
  `PINECONE_NAMESPACE=documentacion-tecnica` (nunca el namespace default
  vacío), para que este dataset no se mezcle con otro que use el mismo
  índice.
- **Chunking en el punto medio (500-800 tokens):** `CHUNK_SIZE_TOKENS=650`
  por defecto, medido en tokens reales con `tiktoken`
  (`from_tiktoken_encoder`), ni tan chico que pierda contexto semántico ni
  tan grande que diluya la precisión del embedding.
- **Texto en la metadata:** cada vector en Pinecone incluye
  `metadata["text"]` con el chunk completo, así el recuperador nunca
  necesita una base de datos relacional aparte para reconstruir el
  contenido a partir de un ID.
- **Metadatos avanzados y trazabilidad:** `source`, `page` (1-indexado
  para PDFs, `None` para Markdown/JSON) y `categoria` viajan intactos
  desde `loaders.py` hasta Pinecone, más un `fuente_id` legible tipo
  `"05_notas_migracion_pydantic_v2.pdf#p2#fragmento_1"`.
- **BM25 no vive en la nube:** `ingest.py` guarda una copia local del
  corpus chunkeado (`corpus_cache.jsonl`, versionado en el repo) en el
  mismo momento en que sube a Pinecone, para que `BM25Retriever` no tenga
  que re-descargar todo el índice.
- **`RAGSystem` testeable sin credenciales:** los retrievers son
  inyectables por constructor, así los tests unitarios (`tests/test_rag_system.py`)
  verifican el ensamblado del `EnsembleRetriever` y el recorte a top-k con
  un retriever vectorial falso, sin gastar llamadas a Pinecone ni a OpenAI.
- **Generación 100% opcional y desacoplada:** `generation.py` no forma
  parte del checklist de esta pre-entrega (que pide recuperación +
  evaluación, no generación). Se agrega como bonus, gratis (Groq/Llama),
  para poder generar evidencia real de una respuesta completa además de
  las métricas. Si no configurás `GROQ_API_KEY`, el resto del sistema
  (ingesta, recuperador híbrido, `evaluate.py`) funciona exactamente
  igual.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

Completá en `.env` (los defaults de `.env.example` ya apuntan a las
opciones gratuitas):

```dotenv
PINECONE_API_KEY=...     # console.pinecone.io -> API Keys (obligatorio)

# Embeddings: gratis por default, no hace falta tocar nada acá.
# EMBEDDINGS_PROVIDER=huggingface

# Generación (opcional/bonus): gratis, sin tarjeta de crédito.
GROQ_API_KEY=...         # console.groq.com/keys
```

### Crear una cuenta y un proyecto de Pinecone (para replicar el índice) — obligatorio

1. Creá una cuenta gratuita en [pinecone.io](https://www.pinecone.io/).
2. En el dashboard, `API Keys` → copiá la key por default a tu `.env`
   (`PINECONE_API_KEY`).
3. **No hace falta crear el índice a mano**: `pinecone_setup.py` lo crea
   solo, en modo Serverless, con la dimensión y región configuradas en
   `.env` (`PINECONE_CLOUD=aws`, `PINECONE_REGION=us-east-1` por defecto,
   que es lo que cubre el free tier de Pinecone). La dimensión se calcula
   sola según el proveedor de embeddings (384 para HuggingFace, el
   default). Si preferís crearlo vos mismo desde la consola: `Create Index`
   → nombre igual a `INDEX_NAME` → `Dimension=384` (o `1536` si usás
   OpenAI) → `Metric=cosine` → `Serverless` → cloud/región a tu elección.

### Conseguir una API key gratis de Groq (opcional — solo si querés probar `test_generacion.py`)

1. Creá una cuenta en [console.groq.com](https://console.groq.com/) (no
   pide tarjeta de crédito).
2. `API Keys` → `Create API Key` → copiala a tu `.env` (`GROQ_API_KEY`).
3. El modelo por default (`llama-3.1-8b-instant`) tiene un free tier con
   límite de requests por minuto/día, más que suficiente para las pruebas
   de este proyecto.

## Uso

### 1. Inicializar el índice (opcional: `ingest.py` ya lo hace por vos)

```bash
python pinecone_setup.py
```

### 2. Ingestar el dataset de ejemplo

```bash
python ingest.py
```

Carga los 5 documentos de `./data` (3 `.md`, 1 `.json`, 1 `.pdf`), los
fragmenta, sube los vectores a Pinecone (namespace
`documentacion-tecnica`) y regenera `corpus_cache.jsonl`. Si el namespace
ya tiene vectores, no vuelve a indexar (`python ingest.py --force` para
forzarlo).

### 3. Consultar el recuperador híbrido

```python
from rag_system import RAGSystem

rag = RAGSystem(top_k=5)
resultados = rag.query("¿Cómo se valida el body de una request en FastAPI?")
for doc in resultados:
    print(doc.metadata["fuente_id"], "->", doc.page_content[:80])
```

### 4. Evaluar (Precision@5 / Recall@5)

```bash
python evaluate.py
```

Corre las 5 preguntas de `golden_set.json` contra `RAGSystem` y calcula:

- **Recall@5**: ¿el documento esperado aparece entre los 5 recuperados?
  (1.0 si sí, 0.0 si no, para golden sets de un solo documento relevante
  por pregunta).
- **Precision@5**: ¿qué proporción de esos 5 fragmentos vienen
  efectivamente del documento esperado?

### 5. (Bonus) Generar una respuesta real de punta a punta

No lo pide el enunciado de esta pre-entrega, pero si querés evidencia de
una respuesta completa (no solo métricas de recuperación) y tenés
`GROQ_API_KEY` configurada:

```bash
python test_generacion.py
```

Corre 3 preguntas contra `RAGSystem.answer()` (recupera con el
`EnsembleRetriever` + genera la respuesta con Groq/Llama): dos
respondibles y una trampa, para confirmar que el LLM dice "no lo sé" en
vez de inventar cuando el contexto recuperado no alcanza.

Ejecucion: 
==============================================================================
[2] Pregunta: ¿Qué servidor ASGI se usa típicamente para correr FastAPI en producción?
==============================================================================
2026-09-13 21:40:54,630 | INFO     | rag_generation | Generando respuesta con Groq (openai/gpt-oss-120b)...
2026-09-13 21:40:57,444 | INFO     | httpx | HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"

Respuesta:
  En producción se suele usar **Uvicorn** como servidor ASGI, típicamente detrás de un gestor de procesos como **Gunicorn** con workers del tipo `uvicorn.workers.UvicornWorker` (o bien varios procesos de Uvicorn detrás de un balanceador).

Fuentes citadas: ['04_faq_fastapi.json#fragmento_0', '04_faq_fastapi.json#fragmento_3', '02_fastapi_dependencias.md#fragmento_0', '03_fastapi_validacion_pydantic.md#fragmento_0', '05_notas_migracion_pydantic_v2.pdf#p1#fragmento_0']

==============================================================================
[3] Pregunta: ¿Cómo se configura el rate limiting con Redis en una aplicación FastAPI?
==============================================================================
2026-09-13 21:40:57,761 | INFO     | rag_generation | Generando respuesta con Groq (openai/gpt-oss-120b)...
2026-09-13 21:40:58,336 | INFO     | httpx | HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"

Respuesta:
  No tengo esa información en los documentos disponibles.

Fuentes citadas: ['04_faq_fastapi.json#fragmento_2', '02_fastapi_dependencias.md#fragmento_0', '03_fastapi_validacion_pydantic.md#fragmento_1', '04_faq_fastapi.json#fragmento_1', '04_faq_fastapi.json#fragmento_3']

Revisá la pregunta [3]: es una trampa (rate limiting con Redis no está en ningún documento del dataset). El modelo debería decir explícitamente que no tiene esa información, no inventarla.
(.venv) lfucc@MacBook-Air preentrega-4-rag-pinecone-hybrid % 


### Sanity check sin Pinecone (solo BM25)

Como parte de la verificación de este entregable, se corrió **solo la
mitad léxica** (BM25, sin tocar Pinecone/OpenAI) contra las 5 preguntas
del golden set, usando el `corpus_cache.jsonl` ya incluido en el repo:

```
[OK] esperado=01_fastapi_routing.md              | top1 = 01_fastapi_routing.md
[OK] esperado=02_fastapi_dependencias.md         | top1 = 02_fastapi_dependencias.md
[OK] esperado=03_fastapi_validacion_pydantic.md  | top1 = 03_fastapi_validacion_pydantic.md
[OK] esperado=04_faq_fastapi.json                | top1 = 04_faq_fastapi.json
[OK] esperado=05_notas_migracion_pydantic_v2.pdf | top1 = 05_notas_migracion_pydantic_v2.pdf
```

Recall@5 = 100% ya con BM25 solo, lo cual confirma que el dataset y el
golden set están bien alineados. **Los números finales de
Precision@5/Recall@5 del sistema híbrido completo dependen de tu propia
cuenta de Pinecone** (los embeddings ya no cuestan nada con el default
HuggingFace) — corré `python evaluate.py` después de `ingest.py` y pegá
acá el resultado real que te imprima la consola:

```
==============================================================================
REPORTE DE EVALUACIÓN — Precision@5 / Recall@5
==============================================================================

[1] Pregunta: ¿Por qué hay que declarar las rutas estáticas antes que las rutas con parámetros variables en FastAPI?
    Documento(s) esperado(s): ['01_fastapi_routing.md']
    Fuentes recuperadas (top-5): ['01_fastapi_routing.md', '02_fastapi_dependencias.md', '04_faq_fastapi.json', '04_faq_fastapi.json', '03_fastapi_validacion_pydantic.md']
    Recall@5:    1.00
    Precision@5: 0.20

[2] Pregunta: ¿Qué ventaja tiene declarar una dependencia con yield en vez de return en FastAPI?
    Documento(s) esperado(s): ['02_fastapi_dependencias.md']
    Fuentes recuperadas (top-5): ['02_fastapi_dependencias.md', '04_faq_fastapi.json', '04_faq_fastapi.json', '05_notas_migracion_pydantic_v2.pdf', '04_faq_fastapi.json']
    Recall@5:    1.00
    Precision@5: 0.20

[3] Pregunta: ¿Por qué conviene no reutilizar el mismo modelo Pydantic para el input y el output de un endpoint?
    Documento(s) esperado(s): ['03_fastapi_validacion_pydantic.md']
    Fuentes recuperadas (top-5): ['03_fastapi_validacion_pydantic.md', '01_fastapi_routing.md', '05_notas_migracion_pydantic_v2.pdf', '03_fastapi_validacion_pydantic.md', '02_fastapi_dependencias.md']
    Recall@5:    1.00
    Precision@5: 0.40

[4] Pregunta: ¿Qué servidor ASGI se usa típicamente para correr una aplicación FastAPI en producción?
    Documento(s) esperado(s): ['04_faq_fastapi.json']
    Fuentes recuperadas (top-5): ['04_faq_fastapi.json', '04_faq_fastapi.json', '02_fastapi_dependencias.md', '05_notas_migracion_pydantic_v2.pdf', '03_fastapi_validacion_pydantic.md']
    Recall@5:    1.00
    Precision@5: 0.40

[5] Pregunta: ¿Cómo se reemplaza la configuración 'class Config' de Pydantic v1 en la versión 2?
    Documento(s) esperado(s): ['05_notas_migracion_pydantic_v2.pdf']
    Fuentes recuperadas (top-5): ['05_notas_migracion_pydantic_v2.pdf', '04_faq_fastapi.json', '02_fastapi_dependencias.md', '01_fastapi_routing.md','05_notas_migracion_pydantic_v2.pdf']
    Recall@5:    1.00
    Precision@5: 0.40

------------------------------------------------------------------------------
RESUMEN
------------------------------------------------------------------------------
Preguntas evaluadas: 5
Recall@5 promedio:    100.00%
Precision@5 promedio: 32.00%
==============================================================================
```

Y, si probaste el paso opcional de generación, pegá acá la salida de
`python test_generacion.py` (respuesta real generada + fuentes citadas,
incluida la pregunta trampa):

```
==============================================================================
[2] Pregunta: ¿Qué servidor ASGI se usa típicamente para correr FastAPI en producción?
==============================================================================
2026-09-13 21:50:35,142 | INFO     | rag_generation | Generando respuesta con Groq (openai/gpt-oss-120b)...
2026-09-13 21:50:35,984 | INFO     | httpx | HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"

Respuesta:
  En producción se suele usar **Uvicorn** como servidor ASGI, típicamente detrás de un gestor de procesos como **Gunicorn** con workers del tipo `uvicorn.workers.UvicornWorker` (o bien varios procesos de Uvicorn detrás de un balanceador).

Fuentes citadas: ['04_faq_fastapi.json#fragmento_0', '04_faq_fastapi.json#fragmento_3', '02_fastapi_dependencias.md#fragmento_0', '03_fastapi_validacion_pydantic.md#fragmento_0', '05_notas_migracion_pydantic_v2.pdf#p1#fragmento_0']

==============================================================================
[3] Pregunta: ¿Cómo se configura el rate limiting con Redis en una aplicación FastAPI?
==============================================================================
2026-09-13 21:50:36,271 | INFO     | rag_generation | Generando respuesta con Groq (openai/gpt-oss-120b)...
2026-09-13 21:50:36,922 | INFO     | httpx | HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"

Respuesta:
  No tengo esa información en los documentos disponibles.

Fuentes citadas: ['04_faq_fastapi.json#fragmento_2', '02_fastapi_dependencias.md#fragmento_0', '03_fastapi_validacion_pydantic.md#fragmento_1', '04_faq_fastapi.json#fragmento_1', '04_faq_fastapi.json#fragmento_3']

```

### Tests unitarios (offline, sin Pinecone, sin OpenAI, sin Groq)

```bash
pytest -v
```

32 tests que verifican: carga multi-formato (`loaders.py`), chunking y
metadata (`ingest.py`), ensamblado del `EnsembleRetriever` con un
retriever vectorial falso (`rag_system.py`), el cableado de
`RAGSystem.answer()` con `generation.py` (con la función de generación
reemplazada por una falsa) y las métricas de evaluación (`evaluate.py`)
— nada de esto requiere credenciales.

## Variables de entorno (`.env`)

| Variable | Descripción | Default |
|---|---|---|
| `PINECONE_API_KEY` | API key de Pinecone | — |
| `PINECONE_CLOUD` / `PINECONE_REGION` | Cloud/región del índice Serverless | `aws` / `us-east-1` |
| `INDEX_NAME` | Nombre del índice de Pinecone | `rag-documentacion-tecnica` |
| `PINECONE_NAMESPACE` | Namespace dentro del índice | `documentacion-tecnica` |
| `EMBEDDINGS_PROVIDER` | `huggingface` (gratis, local) u `openai` (pago) | `huggingface` |
| `HUGGINGFACE_EMBEDDING_MODEL` | Modelo local usado si `EMBEDDINGS_PROVIDER=huggingface` | `sentence-transformers/all-MiniLM-L6-v2` |
| `OPENAI_API_KEY` / `EMBEDDING_MODEL` | Credenciales/modelo si `EMBEDDINGS_PROVIDER=openai` | — / `text-embedding-3-small` |
| `EMBEDDING_DIMENSION` | Dimensión del vector (según proveedor) | `384` (hf) / `1536` (openai) |
| `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` | Chunking (tokens) | `650` / `80` |
| `RAG_TOP_K` | Top-k del recuperador híbrido | `5` |
| `BM25_WEIGHT` / `VECTOR_WEIGHT` | Pesos del `EnsembleRetriever` | `0.5` / `0.5` |
| `GOLDEN_SET_PATH` / `EVAL_TOP_K` | Golden set y top-k de evaluación | `./golden_set.json` / `5` |
| `GROQ_API_KEY` / `GROQ_MODEL` | (Bonus) LLM gratis para `generation.py`/`test_generacion.py` | — / `llama-3.1-8b-instant` |

## Checklist de entrega

- [x] Pipeline de ingesta a Pinecone (`ingest.py` + `pinecone_setup.py`), índice Serverless.
- [x] Metadatos avanzados: fuente, página, categoría (`loaders.py`).
- [x] Texto original guardado en la metadata de Pinecone (`ingest.py`, `metadata["text"]`).
- [x] Namespace explícito (`PINECONE_NAMESPACE`), nunca el default.
- [x] Recuperador híbrido: `BM25Retriever` + `PineconeVectorStore` combinados con `EnsembleRetriever` (`rag_system.py`).
- [x] Clase `RAGSystem` que devuelve top-5 combinando resultados léxicos y semánticos.
- [x] Chunking en el punto medio recomendado (650 tokens, rango 500-800).
- [x] Chequeo de dimensión de embeddings antes de crear/usar el índice (`pinecone_setup.py`).
- [x] `evaluate.py`: Precision@5 y Recall@5 contra un Golden Set de 5 preguntas (`golden_set.json`).
- [x] Reporte impreso en consola (sin PDF ni informe aparte).
- [x] Dataset de ejemplo en 3 formatos: `.md`, `.json`, `.pdf` (documentación técnica de FastAPI/Pydantic).
- [x] Sin API keys hardcodeadas: todo vía `.env` (`.env.example` incluido, `.env` en `.gitignore`).

### Extra (no pedido por el enunciado, agregado para poder probar sin costo)

- [x] Embeddings gratis por default (`EMBEDDINGS_PROVIDER=huggingface`, local, sin API key).
- [x] Paso de generación opcional con LLM gratis (Groq/Llama) — `generation.py`, `RAGSystem.answer()`, `test_generacion.py`.
