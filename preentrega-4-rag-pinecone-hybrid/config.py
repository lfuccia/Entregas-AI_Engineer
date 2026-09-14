"""
config.py
=========

Punto único de configuración (variables de entorno) para todo el
proyecto. Ningún otro módulo lee `os.environ` directamente: todos importan
sus constantes desde acá, para que cambiar un valor (por ejemplo,
`RAG_TOP_K`) no implique buscarlo en cinco archivos distintos.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Pinecone ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
INDEX_NAME = os.getenv("INDEX_NAME", "rag-documentacion-tecnica")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "documentacion-tecnica")
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")

# --- Embeddings ---
# La dimensión del índice de Pinecone tiene que coincidir EXACTAMENTE con
# la dimensión que devuelve el modelo de embeddings elegido. Por default
# se usa "huggingface" (sentence-transformers, LOCAL Y GRATIS, sin API key
# ni tarjeta de crédito) para que todo el pipeline se pueda probar de
# punta a punta sin costo. "openai" queda disponible como alternativa
# (mejor calidad de embeddings, pero de pago) cambiando EMBEDDINGS_PROVIDER.
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "huggingface").strip().lower()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  # si EMBEDDINGS_PROVIDER=openai

HUGGINGFACE_EMBEDDING_MODEL = os.getenv(
    "HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Dimensión del vector que produce el modelo elegido:
#   - sentence-transformers/all-MiniLM-L6-v2 (huggingface, default) -> 384
#   - text-embedding-3-small (openai)                                -> 1536
# Si cambiás de modelo, actualizá EMBEDDING_DIMENSION acorde (y recreá el
# índice de Pinecone si ya existía con otra dimensión: ver pinecone_setup.py).
_DIMENSION_DEFAULT = "384" if EMBEDDINGS_PROVIDER == "huggingface" else "1536"
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", _DIMENSION_DEFAULT))

# --- LLM de generación (opcional: no lo pide el enunciado de esta
# pre-entrega, pero permite generar una respuesta final de verdad —no solo
# métricas de recuperación— como evidencia adicional para el README).
# Groq ofrece modelos open-source (Llama) con un free tier real, sin
# tarjeta de crédito: https://console.groq.com -> API Keys.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# --- Datos / chunking ---
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
# Punto medio entre "pierde contexto" (chunks muy chicos) y "diluye la
# precisión del embedding" (chunks muy grandes): 500-800 tokens.
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "650"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "80"))

# --- Corpus local (cache para BM25 y para no re-consultar Pinecone) ---
CORPUS_CACHE_PATH = Path(os.getenv("CORPUS_CACHE_PATH", "./corpus_cache.jsonl"))

# --- Recuperación híbrida ---
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.5"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.5"))

# --- Evaluación ---
GOLDEN_SET_PATH = Path(os.getenv("GOLDEN_SET_PATH", "./golden_set.json"))
EVAL_TOP_K = int(os.getenv("EVAL_TOP_K", "5"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
