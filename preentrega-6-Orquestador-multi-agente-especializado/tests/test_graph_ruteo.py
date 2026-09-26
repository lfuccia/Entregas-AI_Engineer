"""
tests/test_graph_ruteo.py

Tests offline (sin API key, sin red) del grafo completo: se inyectan LLMs
determinísticos para los tres roles (supervisor, investigador, analista)
vía `construir_grafo(obtener_llm_...=...)`, y se verifica que:

1. El Supervisor rutea correctamente a cada especialista y termina en
   `FINISH` -> `END`.
2. Las `contribuciones` (el "Estado Compartido Estructurado") se van
   acumulando en orden, una por cada paso, sin perder ninguna.
3. El ciclo de refinamiento funciona: si el Supervisor no da por
   suficiente el output del analista, lo vuelve a mandar a "analista" (no
   a investigador ni a FINISH) con una instrucción puntual.
4. Cada especialista recibe SOLO la instrucción puntual del Supervisor
   (más, el analista, el último hallazgo del investigador) — no el
   historial completo de mensajes del sistema.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

import graph as graph_module
from demo_llm import ChatModelDeterminista, SupervisorDeterminista
from graph import construir_grafo
from state import estado_inicial
from supervisor import DecisionSupervisor

RECURSION_LIMIT_TEST = 15


def _investigador_fake(hallazgo: str = "Encontré 2 reseñas: una positiva y una negativa."):
    return ChatModelDeterminista(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "buscar_informacion", "args": {"query": "producto"}, "id": "c1"}],
            ),
            AIMessage(content=hallazgo),
        ]
    )


def _analista_fake(conclusiones: list[str]):
    """Un intento por conclusión: cada uno hace 1 tool call + 1 respuesta final."""
    guion = []
    for i, conclusion in enumerate(conclusiones):
        guion.append(
            AIMessage(
                content="",
                tool_calls=[{"name": "analizar_sentimiento", "args": {"textos": ["x"]}, "id": f"a{i}"}],
            )
        )
        guion.append(AIMessage(content=conclusion))
    return ChatModelDeterminista(responses=guion)


async def test_flujo_completo_investigador_luego_analista_luego_finish():
    supervisor_fake = SupervisorDeterminista(
        [
            DecisionSupervisor(siguiente="investigador", instruccion="Investigá X.", razon="Falta investigar."),
            DecisionSupervisor(siguiente="analista", instruccion="Analizá X.", razon="Falta analizar."),
            DecisionSupervisor(siguiente="FINISH", instruccion="Resumen final.", razon="Ya está todo."),
        ]
    )

    grafo = construir_grafo(
        obtener_llm_supervisor=lambda: supervisor_fake,
        obtener_llm_investigador=lambda: _investigador_fake(),
        obtener_llm_analista=lambda: _analista_fake(["El análisis está completo."]),
    ).compile()

    resultado = await grafo.ainvoke(
        estado_inicial("¿Cómo fue recibido el producto X?"),
        config={"recursion_limit": RECURSION_LIMIT_TEST},
    )

    assert resultado["tarea_completada"] is True
    assert resultado["pasos_dados"] == 3

    agentes_en_orden = [c["agente"] for c in resultado["contribuciones"]]
    assert agentes_en_orden == ["supervisor", "investigador", "supervisor", "analista"]


async def test_ciclo_de_refinamiento_vuelve_al_analista():
    supervisor_fake = SupervisorDeterminista(
        [
            DecisionSupervisor(siguiente="investigador", instruccion="Investigá X.", razon="Falta investigar."),
            DecisionSupervisor(siguiente="analista", instruccion="Analizá X.", razon="Falta analizar."),
            DecisionSupervisor(
                siguiente="analista",
                instruccion="Te faltó calcular el promedio, agregalo.",
                razon="Análisis incompleto: no es suficiente.",
            ),
            DecisionSupervisor(siguiente="FINISH", instruccion="Resumen final.", razon="Ahora sí está completo."),
        ]
    )

    investigador_fake = _investigador_fake()
    analista_fake = _analista_fake(["Solo el sentimiento.", "Sentimiento y promedio."])
    grafo = construir_grafo(
        obtener_llm_supervisor=lambda: supervisor_fake,
        obtener_llm_investigador=lambda: investigador_fake,
        obtener_llm_analista=lambda: analista_fake,
    ).compile()

    resultado = await grafo.ainvoke(
        estado_inicial("¿Cómo fue recibido el producto X?"),
        config={"recursion_limit": RECURSION_LIMIT_TEST},
    )

    assert resultado["pasos_dados"] == 4
    agentes_en_orden = [c["agente"] for c in resultado["contribuciones"]]
    # El analista aparece DOS veces (refinamiento), sin que el investigador
    # tenga que volver a intervenir.
    assert agentes_en_orden == ["supervisor", "investigador", "supervisor", "analista", "supervisor", "analista"]
    assert resultado["contribuciones"][-1]["contenido"] == "Sentimiento y promedio."


def test_especialista_recibe_solo_la_instruccion_puntual_no_todo_el_historial():
    """La pieza central de 'evitar contaminación de contexto': el helper
    que arma la instrucción para cada especialista usa `contribuciones`
    (la última del supervisor), no el historial completo de `messages`."""
    estado = estado_inicial(
        "Una pregunta original larguísima con muchísimo detalle que el "
        "especialista NO debería tener que leer entero."
    )
    estado["contribuciones"] = [{"agente": "supervisor", "contenido": "Instrucción puntual y corta."}]

    instruccion = graph_module._instruccion_para_especialista(estado)
    assert instruccion == "Instrucción puntual y corta."


def test_especialista_usa_la_pregunta_original_si_todavia_no_hay_instruccion_del_supervisor():
    estado = estado_inicial("¿Cuál es la pregunta original?")
    instruccion = graph_module._instruccion_para_especialista(estado)
    assert instruccion == "¿Cuál es la pregunta original?"
