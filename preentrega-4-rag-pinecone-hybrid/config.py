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

# --- Embeddings (OpenAI: la dimensión del índice de Pinecone tiene que
# coincidir EXACTAMENTE con la dimensión que devuelve este modelo) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

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
