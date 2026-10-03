"""
tests/test_graph_ruteo.py

Tests offline (sin API key, sin red) del grafo completo, incluyendo el
nuevo nodo de aprobación humana (HITL) de esta entrega: se inyectan LLMs
determinísticos para los tres roles (supervisor, investigador, analista)
vía `construir_grafo(obtener_llm_...=...)`, más un `InMemorySaver` (el
checkpointer de referencia de LangGraph, 100% en RAM) — hace falta un
checkpointer real para que `interrupt(...)` funcione, pero no hace falta
que sea el `RedisSaver` acá: eso ya se prueba aparte en
`test_redis_checkpointer.py`.

Se verifica que:

1. El Supervisor rutea correctamente a cada especialista y, antes de
   llegar al analista, el grafo se frena en `aprobacion_humana` y hay que
   aprobarlo explícitamente (`Command(resume=...)`) para que siga.
2. Las `contribuciones` (el "Estado Compartido Estructurado") se van
   acumulando en orden, una por cada paso, sin perder ninguna.
3. El ciclo de refinamiento funciona: si el Supervisor no da por
   suficiente el output del analista, lo vuelve a mandar a "analista"
   (pasando otra vez por la aprobación humana) y no a investigador ni a
   FINISH.
4. Cada especialista recibe SOLO la instrucción puntual del Supervisor.
"""

from __future__ import annotations

import uuid

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import app.graph as graph_module
from app.demo_llm import ChatModelDeterminista, SupervisorDeterminista
from app.graph import construir_grafo
from app.state import estado_inicial
from app.supervisor import DecisionSupervisor

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


def _config_nuevo() -> dict:
    return {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": RECURSION_LIMIT_TEST,
    }


async def test_flujo_completo_investigador_luego_hitl_luego_analista_luego_finish():
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
        checkpointer=InMemorySaver(),
    )

    config = _config_nuevo()
    resultado = await grafo.ainvoke(estado_inicial("¿Cómo fue recibido el producto X?"), config=config)

    # El grafo tiene que frenarse ANTES de ejecutar al analista, esperando
    # aprobación humana -- nunca debería llegar a FINISH en esta primera
    # invocación.
    assert "__interrupt__" in resultado
    payload = resultado["__interrupt__"][0].value
    assert payload["tipo"] == "aprobacion_requerida"
    assert payload["accion"] == "analista.calcular_metricas_y_sentimiento"
    assert payload["instruccion_pendiente"] == "Analizá X."

    resultado = await grafo.ainvoke(Command(resume={"aprobado": True, "comentario": "ok"}), config=config)

    assert resultado["tarea_completada"] is True
    assert resultado["pasos_dados"] == 3

    agentes_en_orden = [c["agente"] for c in resultado["contribuciones"]]
    assert agentes_en_orden == ["supervisor", "investigador", "supervisor", "analista"]


async def test_rechazo_de_aprobacion_humana_no_ejecuta_al_analista_y_vuelve_al_supervisor():
    supervisor_fake = SupervisorDeterminista(
        [
            DecisionSupervisor(siguiente="analista", instruccion="Analizá X.", razon="Falta analizar."),
            DecisionSupervisor(
                siguiente="FINISH",
                instruccion="Cierro sin el análisis: fue rechazado por el revisor.",
                razon="La acción crítica fue rechazada.",
            ),
        ]
    )

    def _analista_que_no_deberia_llamarse():
        raise AssertionError(
            "El LLM del analista NO debería invocarse: la aprobación humana fue rechazada."
        )

    grafo = construir_grafo(
        obtener_llm_supervisor=lambda: supervisor_fake,
        obtener_llm_analista=_analista_que_no_deberia_llamarse,
        checkpointer=InMemorySaver(),
    )

    config = _config_nuevo()
    resultado = await grafo.ainvoke(estado_inicial("¿Cómo fue recibido el producto X?"), config=config)
    assert "__interrupt__" in resultado

    resultado = await grafo.ainvoke(
        Command(resume={"aprobado": False, "comentario": "Demasiado costoso por ahora."}),
        config=config,
    )

    assert resultado["tarea_completada"] is True
    agentes_en_orden = [c["agente"] for c in resultado["contribuciones"]]
    assert agentes_en_orden == ["supervisor", "analista"]
    assert "rechaz" in resultado["contribuciones"][-1]["contenido"].lower()
    assert "Demasiado costoso" in resultado["contribuciones"][-1]["contenido"]


async def test_ciclo_de_refinamiento_vuelve_al_analista_pasando_de_nuevo_por_hitl():
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
        checkpointer=InMemorySaver(),
    )

    config = _config_nuevo()
    resultado = await grafo.ainvoke(estado_inicial("¿Cómo fue recibido el producto X?"), config=config)
    assert "__interrupt__" in resultado  # 1er pedido de aprobación (primer paso del analista)

    resultado = await grafo.ainvoke(Command(resume={"aprobado": True}), config=config)
    assert "__interrupt__" in resultado  # 2do pedido de aprobación (refinamiento)

    resultado = await grafo.ainvoke(Command(resume={"aprobado": True}), config=config)

    assert resultado["pasos_dados"] == 4
    agentes_en_orden = [c["agente"] for c in resultado["contribuciones"]]
    assert agentes_en_orden == ["supervisor", "investigador", "supervisor", "analista", "supervisor", "analista"]
    assert resultado["contribuciones"][-1]["contenido"] == "Sentimiento y promedio."


def test_especialista_recibe_solo_la_instruccion_puntual_no_todo_el_historial():
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
