"""
graph.py
========

Arma el `StateGraph` jerárquico completo (heredado de la pre-entrega 6,
con el agregado del nodo de Human-in-the-loop de esta entrega):

                 ┌──────────────┐
        ┌───────▶│  supervisor  │◀────────────────┐
        │        └──────┬───────┘                 │
        │     enrutar_desde_supervisor             │
        │      (Literal: next_agent)               │
        │   ┌─────────┴──────────┐                 │
        │   ▼                    ▼                 │
        │ investigador     aprobacion_humana        │
        │   │               (interrupt + espera     │
        │   │                aprobación externa)     │
        │   │              ┌─────┴──────┐           │
        │   │              ▼            ▼           │
        │   │          analista    (rechazado)       │
        │   │              │            │           │
        └───┴──────────────┴────────────┘           │
                    (FINISH -> END)

- El Supervisor decide, con salida estructurada, quién sigue, igual que
  en la pre-entrega 6.
- Antes de ejecutar al "analista", el grafo pasa por
  `hitl.nodo_aprobacion_humana`, que se congela con `interrupt(...)` y
  espera una aprobación externa real (ver `app/hitl.py` y
  `app/worker.py`). Si se aprueba, sigue a "analista"; si se rechaza,
  vuelve directo al supervisor sin ejecutarlo.
- Cada especialista recibe SOLO la instrucción puntual que dejó el
  Supervisor en `contribuciones` (más, para el analista, el último
  hallazgo del investigador si existe).
"""

from __future__ import annotations

from typing import Any, Callable, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph

from app.agents.analyst_agent import construir_agente_analista
from app.agents.research_agent import construir_agente_investigador
from app.config import AGENTE_RECURSION_LIMIT
from app.hitl import nodo_aprobacion_humana
from app.llm import get_llm
from app.state import Contribucion, EstadoOrquestador
from app.supervisor import construir_nodo_supervisor


def _ultima_contribucion_de(estado: EstadoOrquestador, agente: str) -> str | None:
    for contribucion in reversed(estado["contribuciones"]):
        if contribucion["agente"] == agente:
            return contribucion["contenido"]
    return None


def _pregunta_original(estado: EstadoOrquestador) -> str:
    return next(
        (m.content for m in estado["messages"] if isinstance(m, HumanMessage)),
        "",
    )


def _instruccion_para_especialista(estado: EstadoOrquestador) -> str:
    """La instrucción puntual que dejó el Supervisor, o la pregunta
    original del usuario si todavía no hay ninguna (primer paso)."""
    return _ultima_contribucion_de(estado, "supervisor") or _pregunta_original(estado)


async def _invocar_especialista_acotado(
    agente: Any, contenido_entrada: str, nombre_agente: str
) -> list[BaseMessage]:
    """Invoca el sub-agente ReAct de un especialista con SU PROPIO
    `recursion_limit` (`AGENTE_RECURSION_LIMIT`), independiente del
    `recursion_limit` del grafo principal (ver la explicación larga en
    el README — bug real detectado en la pre-entrega 6 con un LLM real)."""
    try:
        resultado = await agente.ainvoke(
            {"messages": [HumanMessage(content=contenido_entrada)]},
            config={"recursion_limit": AGENTE_RECURSION_LIMIT},
        )
    except GraphRecursionError:
        return [
            AIMessage(
                content=(
                    f"No pude completar la tarea dentro del límite de pasos "
                    f"permitido ({AGENTE_RECURSION_LIMIT}) — puede que haya "
                    "quedado reintentando la misma herramienta. Reporto esto "
                    "en vez de seguir insistiendo."
                ),
                name=nombre_agente,
            )
        ]

    mensajes_del_agente = [
        m.model_copy(update={"name": nombre_agente}) if isinstance(m, AIMessage) and not m.name else m
        for m in resultado["messages"][1:]
    ]
    ultimo = mensajes_del_agente[-1] if mensajes_del_agente else None

    if ultimo is None or (isinstance(ultimo, AIMessage) and ultimo.tool_calls):
        mensajes_del_agente.append(
            AIMessage(
                content=(
                    "No llegué a una respuesta final dentro del límite de "
                    f"pasos permitido ({AGENTE_RECURSION_LIMIT})."
                ),
                name=nombre_agente,
            )
        )

    return mensajes_del_agente


def construir_grafo(
    obtener_llm_supervisor: Callable[[], Any] = get_llm,
    obtener_llm_investigador: Callable[[], Any] = get_llm,
    obtener_llm_analista: Callable[[], Any] = get_llm,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Arma el `StateGraph` y lo compila con el `checkpointer` dado (si
    no se pasa ninguno, LangGraph no persiste nada entre invocaciones —
    suficiente para tests, pero en `app/main.py` SIEMPRE se pasa el
    `RedisSaver` real para que el HITL y la reanudación entre pedidos
    HTTP funcionen). Cada rol recibe su propia factory de LLM
    inyectable, para poder testear/demostrar el ruteo con LLMs falsos
    sin gastar ninguna API real."""

    async def nodo_investigador(estado: EstadoOrquestador) -> dict[str, Any]:
        agente = construir_agente_investigador(obtener_llm_investigador)
        instruccion = _instruccion_para_especialista(estado)

        mensajes_del_agente = await _invocar_especialista_acotado(agente, instruccion, "investigador")
        hallazgo = mensajes_del_agente[-1].content

        contribucion: Contribucion = {"agente": "investigador", "contenido": hallazgo}
        return {"messages": mensajes_del_agente, "contribuciones": [contribucion]}

    async def nodo_analista(estado: EstadoOrquestador) -> dict[str, Any]:
        agente = construir_agente_analista(obtener_llm_analista)
        instruccion = _instruccion_para_especialista(estado)
        datos_investigador = _ultima_contribucion_de(estado, "investigador")

        contenido = instruccion
        if datos_investigador:
            contenido = f"{instruccion}\n\nDatos disponibles (del investigador):\n{datos_investigador}"

        mensajes_del_agente = await _invocar_especialista_acotado(agente, contenido, "analista")
        conclusion = mensajes_del_agente[-1].content

        contribucion: Contribucion = {"agente": "analista", "contenido": conclusion}
        # Se resetea la bandera de aprobación: si el supervisor decide mandar
        # al analista otra vez más adelante (refinamiento), tiene que pasar
        # de nuevo por el nodo de aprobación humana, no arrastrar el "True"
        # de la vez anterior.
        return {"messages": mensajes_del_agente, "contribuciones": [contribucion], "aprobacion_analista": None}

    def enrutar_desde_supervisor(estado: EstadoOrquestador) -> Literal["investigador", "aprobacion_humana", "__end__"]:
        siguiente = estado["next_agent"]
        if siguiente == "FINISH" or siguiente is None:
            return END
        if siguiente == "analista":
            return "aprobacion_humana"
        return siguiente

    def enrutar_desde_aprobacion(estado: EstadoOrquestador) -> Literal["analista", "supervisor"]:
        return "analista" if estado.get("aprobacion_analista") else "supervisor"

    builder = StateGraph(EstadoOrquestador)
    builder.add_node("supervisor", construir_nodo_supervisor(obtener_llm_supervisor))
    builder.add_node("investigador", nodo_investigador)
    builder.add_node("aprobacion_humana", nodo_aprobacion_humana)
    builder.add_node("analista", nodo_analista)

    builder.set_entry_point("supervisor")
    builder.add_conditional_edges(
        "supervisor",
        enrutar_desde_supervisor,
        {"investigador": "investigador", "aprobacion_humana": "aprobacion_humana", END: END},
    )
    builder.add_conditional_edges(
        "aprobacion_humana",
        enrutar_desde_aprobacion,
        {"analista": "analista", "supervisor": "supervisor"},
    )
    builder.add_edge("investigador", "supervisor")
    builder.add_edge("analista", "supervisor")

    return builder.compile(checkpointer=checkpointer)
