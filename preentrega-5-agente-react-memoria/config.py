"""
config.py
=========

Configuración centralizada, leída de variables de entorno (`.env`). Ningún
módulo del proyecto hardcodea una API key: todo pasa por acá.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# -- Proveedor de LLM --------------------------------------------------------
# "groq" (gratis, recomendado para poder probar el agente sin tarjeta),
# "openai" o "anthropic". El grafo (graph.py / llm.py) no cambia según el
# proveedor: solo cambia qué cliente de LangChain se instancia.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

# -- Persistencia (Checkpointer) --------------------------------------------
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "checkpoints.sqlite")

# -- Límite de pasos del grafo (evita bucles infinitos / costos inesperados) -
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "10"))
