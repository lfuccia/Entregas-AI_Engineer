"""
graph.py
========

Define el `StateGraph` del agente ReAct:

    agente --(tools_condition)--> herramientas --> agente --> ... --> END

- Estado: `MessagesState` (built-in de LangGraph). La lista de mensajes se
  acumula sola vía el reducer `add_messages` que trae incorporado — no hace
  falta declarar un TypedDict propio ni un `operator.add` manual.
- Nodo "agente": llama al LLM con las herramientas "bindeadas"
  (`llm.bind_tools(...)`). El propio LLM decide, mirando el prompt del
  usuario y los docstrings de las herramientas, si responde directamente o
  pide ejecutar una herramienta — no hay ningún `if/else` manual de
  enrutamiento.
- Nodo "herramientas": `ToolNode` (prebuilt de LangGraph) ejecuta la
  herramienta que el LLM haya pedido y agrega el resultado como
  `ToolMessage`.
- Arista condicional: `tools_condition` (prebuilt) mira el último mensaje
  del LLM: si tiene `tool_calls`, va a "herramientas"; si no, termina
  (`END`) porque el LLM ya considera que tiene todo lo necesario para
  responder.

`construir_grafo()` acepta `obtener_llm` como inyección de dependencia
(por defecto, `llm.get_llm`, que arma el cliente real de Groq/OpenAI/
Anthropic) para poder testear el ruteo del grafo con un LLM falso y
determinístico, sin gastar llamadas a ninguna API (ver `demo_llm.py` y
`tests/test_graph_ruteo.py`).
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.state import StateGraph as StateGraphType  # solo para el type hint de retorno
from langgraph.prebuilt import ToolNode, tools_condition

from llm import get_llm
from tools import buscar_cliente, buscar_pedidos

HERRAMIENTAS = [buscar_cliente, buscar_pedidos]

PROMPT_SISTEMA = SystemMessage(
    content=(
        "Sos un asistente interno que responde preguntas sobre clientes y "
        "sus pedidos, usando exclusivamente las herramientas disponibles "
        "(nunca inventes cantidades, montos ni fechas). Reglas:\n"
        "1) Si el usuario menciona un cliente por NOMBRE, usá primero "
        "`buscar_cliente` para resolver su ID antes de consultar pedidos.\n"
        "2) Si el usuario ya te dio el ID numérico, andá directo a "
        "`buscar_pedidos`.\n"
        "3) Si una herramienta te devuelve varios candidatos ambiguos, o un "
        "error, NO asumas ni inventes cuál es el correcto: preguntale al "
        "usuario que aclare, o intentá una búsqueda distinta si tenés con "
        "qué.\n"
        "4) Respondé siempre en español, de forma breve y concreta, citando "
        "los números tal cual los devolvió la herramienta."
    )
)


def construir_grafo(obtener_llm: Callable[[], Any] = get_llm) -> StateGraphType:
    """Arma (sin compilar) el `StateGraph` del agente. `obtener_llm` es una
    factory sin argumentos que devuelve el chat model a usar — se puede
    reemplazar en los tests/demo para no depender de una API real."""

    async def nodo_agente(state: MessagesState) -> dict[str, list[BaseMessage]]:
        llm = obtener_llm()
        llm_con_herramientas = llm.bind_tools(HERRAMIENTAS)

        mensajes = state["messages"]
        if not mensajes or not isinstance(mensajes[0], SystemMessage):
            mensajes = [PROMPT_SISTEMA, *mensajes]

        respuesta = await llm_con_herramientas.ainvoke(mensajes)
        return {"messages": [respuesta]}

    builder = StateGraph(MessagesState)
    builder.add_node("agente", nodo_agente)
    builder.add_node("herramientas", ToolNode(HERRAMIENTAS))

    builder.set_entry_point("agente")
    builder.add_conditional_edges(
        "agente",
        tools_condition,
        {"tools": "herramientas", END: END},
    )
    builder.add_edge("herramientas", "agente")

    return builder
