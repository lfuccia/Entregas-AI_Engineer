"""
state.py
========

Esquema de `State` compartido entre el Supervisor y los agentes
especialistas. Hereda de `MessagesState` (que ya trae `messages` con su
reducer `add_messages`) y le suma los campos de control que pide el
enunciado:

- `next_agent`: a quién le toca actuar ahora (o `"FINISH"`). Lo fija el
  Supervisor en cada paso; las aristas condicionales del grafo lo leen
  para rutear. Es un `Literal`, no un string libre, así el tipo mismo
  documenta las únicas rutas válidas del grafo.
- `contribuciones`: lista de aportes de cada especialista
  (`{"agente": ..., "contenido": ...}`), con reducer `operator.add` para
  que se VAYAN ACUMULANDO en vez de pisarse entre pasos. Es la pieza
  central de "Estado Compartido Estructurado": permite que el Supervisor
  (y, si hiciera falta, cualquier otro nodo) sepa exactamente qué agente
  aportó qué información sin tener que releer/parsear todo el historial
  de mensajes, y sin perder ese contexto aunque la conversación crezca.
- `tarea_completada`: lo pone en `True` el Supervisor cuando decide
  `FINISH`. Es más una bandera de trazabilidad/inspección del estado
  final que un control de flujo en sí (el ruteo real lo hace
  `next_agent`).
- `pasos_dados`: contador de pasos del Supervisor. Junto con
  `MAX_PASOS_SUPERVISOR` (ver `config.py`) y el `recursion_limit` del
  grafo, es la defensa contra el "Supervisor Infinito": si se llega al
  techo sin que el LLM elija `FINISH` por su cuenta, se fuerza el cierre.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import MessagesState

NombreAgente = Literal["investigador", "analista", "FINISH"]


class Contribucion(TypedDict):
    agente: str
    contenido: str


class EstadoOrquestador(MessagesState):
    next_agent: NombreAgente | None
    tarea_completada: bool
    contribuciones: Annotated[list[Contribucion], operator.add]
    pasos_dados: int


def estado_inicial(pregunta: str) -> dict:
    """Helper para armar el diccionario de entrada de una invocación nueva
    (`grafo.ainvoke(estado_inicial("..."), config=...)`)."""
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content=pregunta)],
        "next_agent": None,
        "tarea_completada": False,
        "contribuciones": [],
        "pasos_dados": 0,
    }
