"""
tests/test_supervisor.py

Tests offline del nodo Supervisor: la defensa dura contra el "Supervisor
Infinito" (si `pasos_dados` ya llegó a `MAX_PASOS_SUPERVISOR`, se fuerza
`FINISH` SIN llamar al LLM en absoluto — se verifica pasando una factory
de LLM que explota si se la llega a invocar).
"""

from __future__ import annotations

import app.supervisor as supervisor_module
from app.state import estado_inicial


def _llm_que_no_deberia_llamarse():
    raise AssertionError("No debería haberse llamado al LLM: el límite de pasos ya se alcanzó.")


async def test_se_fuerza_finish_al_llegar_al_limite_de_pasos(monkeypatch):
    monkeypatch.setattr(supervisor_module, "MAX_PASOS_SUPERVISOR", 2)

    nodo = supervisor_module.construir_nodo_supervisor(obtener_llm=_llm_que_no_deberia_llamarse)
    estado = estado_inicial("¿algo?")
    estado["pasos_dados"] = 2  # ya en el límite

    resultado = await nodo(estado)

    assert resultado["next_agent"] == "FINISH"
    assert resultado["tarea_completada"] is True
    assert resultado["pasos_dados"] == 3


async def test_decision_normal_delega_al_llm(monkeypatch):
    monkeypatch.setattr(supervisor_module, "MAX_PASOS_SUPERVISOR", 6)

    class LLMFalsoQueDecideInvestigador:
        def with_structured_output(self, esquema):
            return self

        async def ainvoke(self, mensajes):
            return supervisor_module.DecisionSupervisor(
                siguiente="investigador",
                instruccion="Buscá lo que haga falta.",
                razon="Todavía no hay contribuciones.",
            )

    nodo = supervisor_module.construir_nodo_supervisor(obtener_llm=lambda: LLMFalsoQueDecideInvestigador())
    estado = estado_inicial("¿algo?")

    resultado = await nodo(estado)

    assert resultado["next_agent"] == "investigador"
    assert resultado["tarea_completada"] is False
    assert resultado["contribuciones"] == [{"agente": "supervisor", "contenido": "Buscá lo que haga falta."}]
    assert resultado["pasos_dados"] == 1
