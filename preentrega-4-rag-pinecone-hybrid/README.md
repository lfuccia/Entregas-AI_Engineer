# Sistema RAG escalable en la nube con Pinecone

Módulo de Recuperación Escalable: ingesta documentos técnicos en un índice
**Pinecone Serverless**, los recupera con un **Recuperador Híbrido**
(BM25 léxico + similitud vectorial, combinados con `EnsembleRetriever`) y
mide la calidad de esa recuperación con un **script de evaluación**
(Precision@5 / Recall@5) contra un Golden Set de preguntas.

## Componentes

| Archivo | Qué hace |
|---|---|
| `data/` | Dataset de ejemplo: documentación técnica de FastAPI/Pydantic en 3 formatos — `.md`, `.json` y `.pdf` |
| `config.py` | Única fuente de variables de entorno para todo el proyecto |
| `embeddings.py` | Modelo de embeddings (OpenAI `text-embedding-3-small`, 1536 dims), compartido por ingesta y consulta |
| `loaders.py` | Carga multi-formato (.md/.json/.pdf) con metadatos avanzados (fuente, página, categoría) |
| `pinecone_setup.py` | Verifica/crea el índice Pinecone Serverless; detecta mismatch de dimensiones |
| `ingest.py` | Chunking (500-800 tokens) + subida a Pinecone con namespace + metadata + texto original |
| `corpus_cache.py` | Persistencia local del corpus chunkeado (para BM25) |
| `rag_system.py` | Clase `RAGSystem`: `EnsembleRetriever` (BM25 + Pinecone) |
| `evaluate.py` | Precision@5 / Recall@5 contra `golden_set.json`, reporte en consola |
| `golden_set.json` | 5 preguntas con documento fuente conocido |
| `tests/` | 26 tests unitarios offline (sin Pinecone, sin OpenAI) |

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
     └──> embeddings.py (OpenAI text-embedding-3-small, 1536 dims)
              │
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
              ▼
     evaluate.py: Precision@5 / Recall@5 contra golden_set.json
```

### Decisiones clave (y los errores que evitan)

- **Mismatch de dimensiones:** `pinecone_setup.py` valida explícitamente
  que la dimensión del índice existente coincida con
  `EMBEDDING_DIMENSION` (1536 para `text-embedding-3-small`) y corta con un
  error claro (`DimensionMismatchError`) si no coinciden, en vez de dejar
  que cada upsert falle uno por uno.
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

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

Completá en `.env`:

```dotenv
PINECONE_API_KEY=...     # console.pinecone.io -> API Keys
OPENAI_API_KEY=...       # platform.openai.com -> API keys (para embeddings)
INDEX_NAME=rag-documentacion-tecnica
```

### Crear una cuenta y un proyecto de Pinecone (para replicar el índice)

1. Creá una cuenta gratuita en [pinecone.io](https://www.pinecone.io/).
2. En el dashboard, `API Keys` → copiá la key por default a tu `.env`
   (`PINECONE_API_KEY`).
3. **No hace falta crear el índice a mano**: `pinecone_setup.py` lo crea
   solo, en modo Serverless, con la dimensión y región configuradas en
   `.env` (`PINECONE_CLOUD=aws`, `PINECONE_REGION=us-east-1` por defecto,
   que es lo que cubre el free tier de Pinecone). Si preferís crearlo vos
   mismo desde la consola: `Create Index` → nombre igual a `INDEX_NAME` →
   `Dimension=1536` → `Metric=cosine` → `Serverless` → cloud/región a tu
   elección (y actualizá `.env` acorde).

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
cuenta de Pinecone/OpenAI** — corré `python evaluate.py` después de
`ingest.py` y pegá acá el resultado real que te imprima la consola:

```
(pegar acá la salida de `python evaluate.py` una vez que tengas Pinecone/OpenAI configurados)
```

### Tests unitarios (offline, sin Pinecone ni OpenAI)

```bash
pytest -v
```

26 tests que verifican: carga multi-formato (`loaders.py`), chunking y
metadata (`ingest.py`), ensamblado del `EnsembleRetriever` con un
retriever vectorial falso (`rag_system.py`), y las métricas de evaluación
(`evaluate.py`) — nada de esto requiere credenciales.

## Variables de entorno (`.env`)

| Variable | Descripción | Default |
|---|---|---|
| `PINECONE_API_KEY` | API key de Pinecone | — |
| `PINECONE_CLOUD` / `PINECONE_REGION` | Cloud/región del índice Serverless | `aws` / `us-east-1` |
| `INDEX_NAME` | Nombre del índice de Pinecone | `rag-documentacion-tecnica` |
| `PINECONE_NAMESPACE` | Namespace dentro del índice | `documentacion-tecnica` |
| `OPENAI_API_KEY` | API key de OpenAI (embeddings) | — |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSION` | Modelo de embeddings y su dimensión | `text-embedding-3-small` / `1536` |
| `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` | Chunking (tokens) | `650` / `80` |
| `RAG_TOP_K` | Top-k del recuperador híbrido | `5` |
| `BM25_WEIGHT` / `VECTOR_WEIGHT` | Pesos del `EnsembleRetriever` | `0.5` / `0.5` |
| `GOLDEN_SET_PATH` / `EVAL_TOP_K` | Golden set y top-k de evaluación | `./golden_set.json` / `5` |

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
