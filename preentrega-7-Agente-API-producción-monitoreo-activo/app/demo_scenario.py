"""
demo_scenario.py
=================

Un escenario de guion fijo (Supervisor/Investigador/Analista
determinísticos, ver `app/demo_llm.py`) empaquetado como una factory de
grafo (`construir_grafo_demo(checkpointer)`) lista para reemplazar a
`app.graph.construir_grafo` cuando `DEMO_MODE=true` (ver `.env.example` y
`app/main.py`) o al correr `scripts/demo_offline.py`.

Por qué una factory y no un grafo ya armado: cada invocación nueva
necesita SU PROPIO guion con el índice en cero (`SupervisorDeterminista`
y `ChatModelDeterminista` son objetos CON ESTADO — van consumiendo su
guion en orden), así que si dos jobs corrieran en paralelo compartiendo
la misma instancia, uno pisaría el guion del otro. `construir_grafo_demo`
arma un escenario nuevo en cada llamada, para que n jobs concurrentes (ver
`scripts/load_test.py`) cada uno tenga el suyo propio, sin interferencia
entre ellos.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from app.demo_llm import ChatModelDeterminista
from app.graph import construir_grafo
from app.supervisor import DecisionSupervisor


class _SupervisorDemoSinEstado:
    """Un Supervisor de guion, pero SIN ESTADO (no es un índice que avanza
    con cada llamada): decide mirando el CONTENIDO del contexto que se le
    pasa (qué contribuciones ya existen), igual que haría un LLM real.

    Esto importa específicamente por el HITL de esta entrega: entre
    `POST /tasks` y `POST /tasks/{id}/approve` el grafo se reconstruye
    desde cero (`construir_grafo_demo(checkpointer)` se llama de nuevo en
    cada request HTTP) -- con un LLM real esto no es un problema, porque
    un LLM real no "recuerda" en qué paso de un guion interno estaba, mira
    la conversación actual y ya. Pero un fake con ÍNDICE interno
    (`SupervisorDeterminista`, usado en los tests, donde todo corre en una
    sola invocación sin reconstruir el grafo en el medio) SÍ "olvidaría"
    cuánto había avanzado si se lo reconstruye a mitad de camino, y
    volvería a repetir su primera decisión en vez de seguir donde iba.
    Por eso el demo de punta a punta con HITL necesita este, no aquel."""

    def with_structured_output(self, esquema):
        return self

    async def ainvoke(self, mensajes) -> DecisionSupervisor:
        contexto = mensajes[-1].content if mensajes else ""
        ya_investigo = "[investigador]:" in contexto
        ya_analizo = "[analista]:" in contexto

        if not ya_investigo:
            return DecisionSupervisor(
                siguiente="investigador",
                instruccion="Buscá reseñas del auricular Aurora X2.",
                razon="Hace falta investigar antes de analizar.",
            )
        if not ya_analizo:
            return DecisionSupervisor(
                siguiente="analista",
                instruccion="Analizá el sentimiento y calculá el promedio de puntaje de lo encontrado.",
                razon="Ya hay datos del investigador.",
            )
        return DecisionSupervisor(
            siguiente="FINISH",
            instruccion=(
                "Las reseñas del Aurora X2 son mayormente positivas (sentimiento positivo), "
                "con un puntaje promedio de 4.2/5."
            ),
            razon="El análisis cubre sentimiento y promedio, como pedía la consulta original.",
        )


def _armar_llms_de_guion():
    supervisor_fake = _SupervisorDemoSinEstado()
    investigador_fake = ChatModelDeterminista(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "buscar_informacion", "args": {"query": "Aurora X2"}, "id": "inv1"}],
            ),
            AIMessage(
                content=(
                    "Encontré 5 reseñas del Aurora X2: 4 positivas (batería, sonido) y 1 negativa "
                    "(estuche). Puntajes: 5, 4, 4, 5, 3."
                )
            ),
        ]
    )
    analista_fake = ChatModelDeterminista(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "analizar_sentimiento",
                        "args": {"textos": ["batería excelente", "sonido excelente", "estuche defectuoso"]},
                        "id": "an1",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "calcular_metricas", "args": {"numeros": [5, 4, 4, 5, 3]}, "id": "an2"}],
            ),
            AIMessage(content="Sentimiento general positivo. Promedio de puntaje: 4.2/5 (sobre 5 reseñas)."),
        ]
    )
    return supervisor_fake, investigador_fake, analista_fake


def construir_grafo_demo(checkpointer=None):
    supervisor_fake, investigador_fake, analista_fake = _armar_llms_de_guion()
    return construir_grafo(
        obtener_llm_supervisor=lambda: supervisor_fake,
        obtener_llm_investigador=lambda: investigador_fake,
        obtener_llm_analista=lambda: analista_fake,
        checkpointer=checkpointer,
    )
