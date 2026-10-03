"""
hitl.py
=======

Nodo de aprobación humana (Human-in-the-loop) del grafo.

Se inserta ENTRE el Supervisor y el nodo "analista": cuando el
Supervisor decide que le toca actuar al analista, el grafo pasa primero
por `nodo_aprobacion_humana`, que llama a `interrupt(...)` y SE
CONGELA ahí mismo (LangGraph guarda el checkpoint completo en Redis vía
`RedisSaver` y la llamada a `.ainvoke()`/`.astream()` en curso retorna,
sin excepción, con una clave `"__interrupt__"` en el resultado).

Por qué el analista y no el investigador
-----------------------------------------
Se modela al analista como la "acción crítica" del sistema: en un caso
real, el paso que agrupa operaciones costosas (más tokens/llamadas al
LLM, cómputo) o con efectos secundarios (escribir a una base de datos,
mandar un mail, ejecutar una orden) es justamente el que conviene
frenar para que un humano lo revise antes de que se ejecute — el
investigador, en cambio, solo lee de un corpus de solo-lectura, así que
no hace falta frenarlo.

Cómo se retoma
---------------
`app/worker.py` detecta la interrupción (`"__interrupt__" in resultado`)
y dejar el job en Redis con status `AWAITING_APPROVAL` más el `payload`
del interrupt. El endpoint `POST /tasks/{id}/approve` (`app/main.py`)
llama de nuevo a `grafo.ainvoke(Command(resume={"aprobado": ..., "comentario": ...}), config=...)`
con el MISMO `thread_id` — LangGraph usa el checkpoint guardado en Redis
para reconstruir el estado exacto en el que se había quedado y
`interrupt(...)` devuelve, en ese re-ingreso al nodo, el valor que se
mandó en `Command(resume=...)`.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from app.state import Contribucion, EstadoOrquestador


def _instruccion_pendiente(estado: EstadoOrquestador) -> str:
    for contribucion in reversed(estado["contribuciones"]):
        if contribucion["agente"] == "supervisor":
            return contribucion["contenido"]
    return ""


async def nodo_aprobacion_humana(estado: EstadoOrquestador) -> dict[str, Any]:
    instruccion = _instruccion_pendiente(estado)

    # Nada de efectos secundarios ANTES de `interrupt(...)`: si el nodo
    # se re-ejecuta al retomar, todo lo anterior a esta línea se vuelve a
    # correr, pero esta línea es lo primero que hace el nodo, así que no
    # hay nada que duplicar.
    decision: dict[str, Any] = interrupt(
        {
            "tipo": "aprobacion_requerida",
            "accion": "analista.calcular_metricas_y_sentimiento",
            "motivo": (
                "Este paso agrupa el cómputo/costo del Agente Analista "
                "(en un sistema real: la acción con efectos secundarios o "
                "de mayor costo del flujo) y requiere aprobación humana "
                "explícita antes de ejecutarse."
            ),
            "instruccion_pendiente": instruccion,
        }
    )

    aprobado = bool(decision.get("aprobado")) if isinstance(decision, dict) else bool(decision)
    comentario = decision.get("comentario", "") if isinstance(decision, dict) else ""

    if aprobado:
        mensaje = AIMessage(
            content=f"Aprobación humana recibida para el analista. {('Comentario: ' + comentario) if comentario else ''}".strip(),
            name="hitl",
        )
        return {"messages": [mensaje], "aprobacion_analista": True}

    motivo_rechazo = comentario or "Rechazado por el revisor humano, sin comentario adicional."
    mensaje = AIMessage(
        content=f"Acción del analista RECHAZADA por aprobación humana: {motivo_rechazo}",
        name="hitl",
    )
    contribucion: Contribucion = {
        "agente": "analista",
        "contenido": f"(No se ejecutó: la aprobación humana fue rechazada. Motivo: {motivo_rechazo})",
    }
    return {
        "messages": [mensaje],
        "aprobacion_analista": False,
        "contribuciones": [contribucion],
    }
