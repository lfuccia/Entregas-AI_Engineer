"""
config.py
=========

Configuración centralizada, leída de variables de entorno (`.env`).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# -- Proveedor de LLM ---------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

# -- Búsqueda del Agente de Investigación --------------------------------
# "simulado" (default, gratis, sin red — busca sobre un corpus propio en
# memoria) o "tavily" (búsqueda web real, requiere TAVILY_API_KEY).
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "simulado").strip().lower()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# -- Control de flujo del Supervisor -------------------------------------
# Techo defensivo de pasos del supervisor (además del recursion_limit del
# grafo): si se llega a este número de decisiones sin llegar a FINISH, se
# fuerza el cierre igual, para evitar el "Supervisor Infinito".
MAX_PASOS_SUPERVISOR = int(os.getenv("MAX_PASOS_SUPERVISOR", "6"))

# `recursion_limit` del grafo PRINCIPAL (supervisor <-> especialistas).
# OJO: LangGraph propaga este mismo presupuesto a los sub-agentes ReAct
# internos de cada especialista (create_react_agent) cuando se invocan sin
# pasarles un config propio — por eso este número tiene que ser generoso:
# con un LLM real, cada especialista puede necesitar varias idas y vueltas
# de tool-calling antes de responder, y esos pasos también consumen este
# mismo presupuesto. Ver AGENTE_RECURSION_LIMIT más abajo para el techo
# INDEPENDIENTE de cada especialista.
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "40"))

# Techo de pasos PROPIO de cada sub-agente ReAct (investigador/analista),
# pasado explícitamente en su propio `config` al invocarlo — así un
# especialista que se cuelga reintentando la misma herramienta falla
# rápido y de forma acotada (ver graph.py), en vez de consumir en
# silencio todo el presupuesto del grafo principal.
AGENTE_RECURSION_LIMIT = int(os.getenv("AGENTE_RECURSION_LIMIT", "10"))
