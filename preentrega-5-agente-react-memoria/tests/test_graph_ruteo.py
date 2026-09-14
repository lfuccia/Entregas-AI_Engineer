"""
tests/test_graph_ruteo.py

Tests offline (sin API key, sin red) del RUTEO del grafo: se inyecta un LLM
falso y determinístico (`demo_llm.LLMDeterministaDeDemo`) vía
`construir_grafo(obtener_llm=...)` para verificar que:

1. Cuando el LLM pide una herramienta, `tools_condition` deriva a
   "herramientas", el `ToolNode` la ejecuta de verdad (contra los datos
   reales de `tools.py`) y el resultado vuelve al nodo "agente".
2. Cuando el LLM responde sin `tool_calls`, el grafo termina (`END`).
3. La persistencia funciona: dos invocaciones con el mismo `thread_id`
   comparten historial (el estado crece), y un `thread_id` distinto arranca
   de cero.

Se usa `AsyncSqliteSaver.from_conn_string(":memory:")` — el checkpointer
SQLite real, pero en memoria, para no ensuciar el disco al testear.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from demo_llm import LLMDeterministaDeDemo
from graph import construir_grafo

RECURSION_LIMIT_TEST = 10


async def test_el_grafo_ejecuta_la_herramienta_pedida_y_responde():
    guion = [
        AIMessage(
            content="",
            tool_calls=[{"name": "buscar_pedidos", "args": {"cliente_id": 102}, "id": "call_1"}],
        ),
        AIMessage(content="El cliente 102 tuvo 3 pedidos por un total de $14.500."),
    ]
    modelo_falso = LLMDeterministaDeDemo(guion)

    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        grafo = construir_grafo(obtener_llm=lambda: modelo_falso).compile(checkpointer=saver)
        cfg = {"configurable": {"thread_id": "t-ruteo"}, "recursion_limit": RECURSION_LIMIT_TEST}

        resultado = await grafo.ainvoke(
            {"messages": [HumanMessage(content="¿Cuántos pedidos tuvo el cliente 102?")]},
            config=cfg,
        )

        tipos = [type(m).__name__ for m in resultado["messages"]]
        assert tipos == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
        assert "14.500" in resultado["messages"][-1].content
        assert resultado["messages"][-1].tool_calls == []

        # El ToolMessage tiene que traer el resultado REAL de la herramienta,
        # no un valor inventado por el guion.
        mensaje_herramienta = resultado["messages"][2]
        assert '"pedidos": 3' in mensaje_herramienta.content


async def test_el_grafo_encadena_dos_herramientas_distintas():
    guion = [
        AIMessage(
            content="",
            tool_calls=[{"name": "buscar_cliente", "args": {"nombre_o_id": "Martina Gómez"}, "id": "c1"}],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "buscar_pedidos", "args": {"cliente_id": 103}, "id": "c2"}],
        ),
        AIMessage(content="Martina Gómez (cliente 103) tuvo 7 pedidos por un total de $58.900."),
    ]
    modelo_falso = LLMDeterministaDeDemo(guion)

    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        grafo = construir_grafo(obtener_llm=lambda: modelo_falso).compile(checkpointer=saver)
        cfg = {"configurable": {"thread_id": "t-multipaso"}, "recursion_limit": RECURSION_LIMIT_TEST}

        resultado = await grafo.ainvoke(
            {"messages": [HumanMessage(content="¿Cuántos pedidos tuvo Martina Gómez?")]},
            config=cfg,
        )

        llamadas_a_herramientas = [
            tc["name"]
            for m in resultado["messages"]
            if isinstance(m, AIMessage)
            for tc in (m.tool_calls or [])
        ]
        assert llamadas_a_herramientas == ["buscar_cliente", "buscar_pedidos"]


async def test_misma_thread_id_recuerda_el_historial():
    guion = [
        AIMessage(
            content="",
            tool_calls=[{"name": "buscar_pedidos", "args": {"cliente_id": 102}, "id": "call_1"}],
        ),
        AIMessage(content="El cliente 102 tuvo 3 pedidos por un total de $14.500."),
        AIMessage(content="El último pedido fue el P-2044."),
    ]
    modelo_falso = LLMDeterministaDeDemo(guion)

    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        grafo = construir_grafo(obtener_llm=lambda: modelo_falso).compile(checkpointer=saver)
        cfg = {"configurable": {"thread_id": "t-memoria"}, "recursion_limit": RECURSION_LIMIT_TEST}

        await grafo.ainvoke(
            {"messages": [HumanMessage(content="¿Cuántos pedidos tuvo el cliente 102?")]},
            config=cfg,
        )
        resultado = await grafo.ainvoke(
            {"messages": [HumanMessage(content="¿Y el último pedido?")]}, config=cfg
        )

        # El historial acumulado tiene los 2 turnos completos: 2 humanos +
        # 1 AIMessage con tool_call + 1 ToolMessage + 2 respuestas finales.
        assert len(resultado["messages"]) == 6


async def test_distinta_thread_id_no_comparte_historial():
    guion_thread_1 = [AIMessage(content="Hola, thread 1.")]
    guion_thread_2 = [AIMessage(content="Hola, thread 2.")]

    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        grafo_1 = construir_grafo(obtener_llm=lambda: LLMDeterministaDeDemo(guion_thread_1)).compile(
            checkpointer=saver
        )
        cfg_1 = {"configurable": {"thread_id": "thread-1"}, "recursion_limit": RECURSION_LIMIT_TEST}
        resultado_1 = await grafo_1.ainvoke({"messages": [HumanMessage(content="Hola")]}, config=cfg_1)
        assert len(resultado_1["messages"]) == 2  # 1 human + 1 AI, sin historial previo

        grafo_2 = construir_grafo(obtener_llm=lambda: LLMDeterministaDeDemo(guion_thread_2)).compile(
            checkpointer=saver
        )
        cfg_2 = {"configurable": {"thread_id": "thread-2"}, "recursion_limit": RECURSION_LIMIT_TEST}
        resultado_2 = await grafo_2.ainvoke({"messages": [HumanMessage(content="Hola")]}, config=cfg_2)
        assert len(resultado_2["messages"]) == 2  # arranca de cero, no ve nada de thread-1
