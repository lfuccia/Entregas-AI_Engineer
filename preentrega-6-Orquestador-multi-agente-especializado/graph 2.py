"""
graph.py
========

Arma el `StateGraph` jerárquico completo:

                 ┌──────────────┐
        ┌───────▶│  supervisor  │◀───────┐
        │        └──────┬───────┘        │
        │     enrutar_desde_supervisor    │
        │      (Literal: next_agent)      │
        │        ┌────┴────┐              │
        │        ▼         ▼              │
        │  investigador  analista         │
        │        │         │              │
        └────────┘         └──────────────┘
                    (FINISH -> END)

- El Supervisor (`supervisor.construir_nodo_supervisor`) decide, con
  salida estructurada, quién sigue: vuelve SIEMPRE al supervisor después
  de cada especialista, así puede evaluar el resultado contra su rúbrica
  antes de decidir si hace falta refinar o ya se puede cerrar.
- Cada especialista recibe SOLO la instrucción puntual que dejó el
  Supervisor en `contribuciones` (más, para el analista, el último
  hallazgo del investigador si existe) — nunca el historial completo de
  mensajes del sistema. Esto es deliberado (ver "Contaminación de
  Contexto" en el enunciado): evita que cada especialista tenga que leer
  y descartar mensajes de otros roles que no le sirven, y mantiene sus
  prompts cortos y baratos.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph

from agents.analyst_agent import construir_agente_analista
from agents.research_agent import construir_agente_investigador
from config import AGENTE_RECURSION_LIMIT
from llm import get_llm
from state import Contribucion, EstadoOrquestador
from supervisor import construir_nodo_supervisor


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
    `recursion_limit` del grafo principal.

    Esto importa porque LangGraph propaga el `recursion_limit` del grafo
    que invoca a un sub-agente cuando este se llama sin pasarle un config
    propio — así que, sin este límite explícito, un especialista que se
    cuelga reintentando la misma herramienta (un LLM real puede hacerlo,
    a diferencia de los LLMs de prueba determinísticos de `demo_llm.py`)
    terminaría consumiendo TODO el presupuesto de pasos del grafo
    principal, y el `StateGraph` entero explotaría con un
    `GraphRecursionError` sin que el Supervisor llegue siquiera a
    enterarse. Acotarlo acá, por especialista, hace que una falla así
    quede contenida: se atrapa el error (o el caso, más sutil, en que el
    límite se alcanza sin excepción pero el agente se quedó a mitad de una
    llamada a herramienta) y se lo convierte en una contribución explícita
    de "no pude terminar" — el Supervisor la va a leer como información
    incompleta y, según su rúbrica, va a reintentar con otra instrucción o
    cerrar explicando la limitación, en vez de que todo el proceso
    reviente.
    """
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
        # Se alcanzó el límite de pasos SIN que LangGraph llegue a lanzar
        # GraphRecursionError (puede pasar justo en el borde del límite):
        # el sub-agente quedó a mitad de una llamada a herramienta, sin
        # respuesta final. Es el mismo caso que el except de arriba, así
        # que se trata igual en vez de devolver contenido vacío/incompleto.
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
) -> StateGraph:
    """Arma (sin compilar) el `StateGraph`. Cada rol recibe su propia
    factory de LLM inyectable, para poder testear/demostrar el ruteo con
    LLMs falsos independientes por rol sin gastar ninguna API real."""

    async def nodo_investigador(estado: EstadoOrquestador) -> dict[str, Any]:
        agente = construir_agente_investigador(obtener_llm_investigador)
        instruccion = _instruccion_para_especialista(estado)

        # `resultado["messages"][0]` (el HumanMessage con la instrucción que
        # nosotros mismos mandamos) ya se descarta adentro del helper: no
        # hace falta duplicarlo en el estado global. Lo que sigue (tool
        # calls, ToolMessages, conclusión) SÍ se agrega — no para que otros
        # nodos lo lean como contexto de entrada (eso sigue viniendo solo de
        # `contribuciones`), sino para que quede un rastro completo y
        # verificable de qué herramienta se usó y qué devolvió.
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
        return {"messages": mensajes_del_agente, "contribuciones": [contribucion]}

    def enrutar_desde_supervisor(estado: EstadoOrquestador) -> Literal["investigador", "analista", "__end__"]:
        siguiente = estado["next_agent"]
        if siguiente == "FINISH" or siguiente is None:
            return END
        return siguiente

    builder = StateGraph(EstadoOrquestador)
    builder.add_node("supervisor", construir_nodo_supervisor(obtener_llm_supervisor))
    builder.add_node("investigador", nodo_investigador)
    builder.add_node("analista", nodo_analista)

    builder.set_entry_point("supervisor")
    builder.add_conditional_edges(
        "supervisor",
        enrutar_desde_supervisor,
        {"investigador": "investigador", "analista": "analista", END: END},
    )
    builder.add_edge("investigador", "supervisor")
    builder.add_edge("analista", "supervisor")

    return builder
