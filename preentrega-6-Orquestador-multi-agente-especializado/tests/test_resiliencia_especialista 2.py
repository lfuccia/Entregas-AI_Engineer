"""
tests/test_resiliencia_especialista.py

Reproduce, en chico y de forma controlada, el bug real detectado al correr
el orquestador con un LLM de verdad: un especialista que queda
reintentando la misma herramienta sin parar puede, si no se lo acota con
su PROPIO `recursion_limit`, consumir todo el presupuesto de pasos del
grafo PRINCIPAL (porque LangGraph se lo propaga a los sub-agentes ReAct
invocados sin un config propio) y hacer explotar el `StateGraph` entero
con un `GraphRecursionError`, sin que el Supervisor llegue siquiera a
enterarse.

Este test verifica que `_invocar_especialista_acotado` (en `graph.py`)
contiene esa falla: con un LLM que SIEMPRE pide la misma herramienta
(nunca da una respuesta final), el nodo "investigador" no revienta —
devuelve una contribución explícita de "no pude completar" dentro de su
propio límite acotado (`AGENTE_RECURSION_LIMIT`), y el grafo principal
sigue funcionando con normalidad (vuelve al supervisor).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import graph as graph_module
from demo_llm import ChatModelDeterminista, SupervisorDeterminista
from graph import construir_grafo
from state import estado_inicial
from supervisor import DecisionSupervisor


class LLMQueSiempreLlamaLaMismaHerramienta(ChatModelDeterminista):
    """Simula un LLM real "colgado": nunca deja de pedir la misma
    herramienta, así que su sub-agente ReAct nunca llega a una respuesta
    final por sí solo."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        mensaje = AIMessage(
            content="",
            tool_calls=[{"name": "buscar_informacion", "args": {"query": "x"}, "id": "c"}],
        )
        return ChatResult(generations=[ChatGeneration(message=mensaje)])


async def test_investigador_colgado_no_hace_explotar_el_grafo_principal(monkeypatch):
    monkeypatch.setattr(graph_module, "AGENTE_RECURSION_LIMIT", 4)

    supervisor_fake = SupervisorDeterminista(
        [
            DecisionSupervisor(
                siguiente="investigador", instruccion="Investigá X.", razon="Falta investigar."
            ),
            DecisionSupervisor(
                siguiente="FINISH",
                instruccion="No se pudo completar la investigación; cierro informando la limitación.",
                razon="El investigador no pudo completar la tarea; no tiene sentido seguir insistiendo.",
            ),
        ]
    )
    investigador_colgado = LLMQueSiempreLlamaLaMismaHerramienta(responses=[])

    grafo = construir_grafo(
        obtener_llm_supervisor=lambda: supervisor_fake,
        obtener_llm_investigador=lambda: investigador_colgado,
    ).compile()

    # El grafo PRINCIPAL no debería necesitar más que un puñado de pasos
    # (supervisor -> investigador -> supervisor -> FINISH), muy por debajo
    # del límite acotado del especialista.
    resultado = await grafo.ainvoke(
        estado_inicial("¿Qué opinan del producto X?"),
        config={"recursion_limit": 15},
    )

    # Lo esencial: el grafo PRINCIPAL no revienta (ni con GraphRecursionError
    # ni con ningún otro error) y llega a FINISH con una contribución del
    # investigador no vacía. El mensaje exacto puede venir de nuestro propio
    # manejo de GraphRecursionError ("no pude completar...") o del propio
    # mecanismo interno de `create_react_agent` para cuando se queda sin
    # pasos (un canned "Sorry, need more steps..." en vez de explotar) — las
    # dos son señales válidas de que el límite acotado funcionó.
    assert resultado["tarea_completada"] is True
    contribucion_investigador = next(c for c in resultado["contribuciones"] if c["agente"] == "investigador")
    contenido = contribucion_investigador["contenido"].lower()
    assert contenido.strip()
    assert "no pude" in contenido or "more steps" in contenido or "más pasos" in contenido
