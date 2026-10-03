"""
supervisor.py
=============

El nodo Supervisor: el router inteligente y controlador de flujo de la
topología jerárquica. En cada paso mira la conversación y las
`contribuciones` acumuladas en el estado y decide, con salida
ESTRUCTURADA (no parseando texto libre), a quién le toca actuar ahora:
`"investigador"`, `"analista"`, o `"FINISH"` si la tarea ya está resuelta.

Usar `with_structured_output` con un `Literal` adentro del schema (en vez
de pedirle al LLM que escriba "investigador" en un texto y parsearlo) es
lo que el enunciado pide con "define... usando Literal en el retorno para
las aristas condicionales": la arista condicional (`enrutar_desde_supervisor`
en `graph.py`) simplemente lee `state["next_agent"]`, que solo puede tomar
esos tres valores por construcción del tipo — no hay forma de que el LLM
devuelva un valor inválido que rompa el ruteo.
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import MAX_PASOS_SUPERVISOR
from app.llm import get_llm
from app.state import Contribucion, EstadoOrquestador, NombreAgente

PROMPT_SUPERVISOR = """Sos el Supervisor de un equipo de dos especialistas que responden \
consultas de análisis e investigación:

- "investigador": busca información externa con su herramienta de búsqueda. ESA \
herramienta consulta un corpus PROPIO y CHICO de reseñas ya cargado de antemano (no \
busca en la web real, no puede ir a buscar a Amazon, Best Buy, foros, etc.) — te va a \
devolver como máximo un puñado de resultados (a veces 3, a veces 5), nunca decenas. \
Eso es esperable y ES SUFICIENTE: NUNCA le pidas al investigador una cantidad mínima \
de reseñas/comentarios (ni "al menos 8", ni ninguna otra cifra inventada por vos) ni \
que busque en sitios externos específicos — con lo que la herramienta devuelva alcanza.
- "analista": procesa datos ya obtenidos (sentimiento, métricas numéricas) con sus herramientas.

Tu trabajo es decidir, en cada paso, quién debe actuar ahora, o si la tarea ya está \
resuelta y hay que finalizar. Mirá la pregunta original del usuario y las \
contribuciones ya hechas por cada especialista (te las paso abajo) para decidir.

Rúbrica para dar por SUFICIENTE la tarea (elegir "FINISH"):
1. Tiene que haber al menos una contribución del investigador con datos concretos \
(no vacía, no un error) — lo que haya traído la búsqueda alcanza, no exijas más.
2. Tiene que haber al menos una contribución del analista que procese ESOS datos \
concretamente (ej. si el pedido incluye tanto sentimiento como una cifra numérica, \
el análisis tiene que cubrir ambas cosas, no solo una).
3. Si el analista respondió pero dejó afuera algo que el pedido original pedía \
explícitamente (ej. calculó el promedio pero no analizó el sentimiento, o al revés), \
NO es suficiente: hay que volver a mandarlo a "analista" con una instrucción puntual \
de qué le faltó (esto es un refinamiento, no un error — es normal y esperado).
4. Si el investigador no trajo nada útil, dale COMO MUCHO una segunda oportunidad con \
una instrucción distinta (ej. otros términos de búsqueda) — nunca una tercera vez con \
el mismo pedido. Si sigue sin haber nada útil, no insistas más: pasá igual al analista \
con lo que haya, o finalizá explicando la limitación.

Nunca inventes vos los datos ni el análisis — tu trabajo es solo decidir el próximo \
paso y, opcionalmente, si el paso es para el analista o el investigador, aclarar qué \
le faltó (sin inventarle requisitos que la herramienta disponible no puede cumplir).
"""


class DecisionSupervisor(BaseModel):
    siguiente: NombreAgente = Field(
        description=(
            "A quién le toca actuar ahora: 'investigador', 'analista', o "
            "'FINISH' si la tarea ya está resuelta según la rúbrica."
        )
    )
    instruccion: str = Field(
        description=(
            "Instrucción puntual y autocontenida para el especialista elegido "
            "(qué tiene que hacer concretamente). Si siguiente='FINISH', una "
            "síntesis final breve para el usuario en su lugar."
        )
    )
    razon: str = Field(description="Por qué se tomó esta decisión (para trazabilidad/log).")


def _resumen_contribuciones(estado: EstadoOrquestador) -> str:
    if not estado["contribuciones"]:
        return "(todavía no hay ninguna contribución de los especialistas)"
    lineas = [f"- [{c['agente']}]: {c['contenido']}" for c in estado["contribuciones"]]
    return "\n".join(lineas)


def _pregunta_original(estado: EstadoOrquestador) -> str:
    primer_mensaje_humano = next(
        (m.content for m in estado["messages"] if m.__class__.__name__ == "HumanMessage"),
        "",
    )
    return primer_mensaje_humano


def construir_nodo_supervisor(obtener_llm: Callable[[], Any] = get_llm):
    """Devuelve la función de nodo `supervisor(state)` para el grafo,
    con `obtener_llm` inyectable (para tests/demo con un LLM falso que
    implemente `.with_structured_output(...)`)."""

    async def nodo_supervisor(estado: EstadoOrquestador) -> dict:
        pasos_dados = estado["pasos_dados"]

        # Defensa dura contra el "Supervisor Infinito": si ya se gastaron
        # todos los pasos permitidos, se fuerza el cierre sin ni siquiera
        # consultar al LLM.
        if pasos_dados >= MAX_PASOS_SUPERVISOR:
            mensaje_limite = (
                "Se alcanzó el límite de pasos del Supervisor "
                f"({MAX_PASOS_SUPERVISOR}) antes de que la tarea se marcara "
                "como completa. Cierro con lo que hay disponible hasta ahora."
            )
            return {
                "messages": [AIMessage(content=mensaje_limite, name="supervisor")],
                "next_agent": "FINISH",
                "tarea_completada": True,
                "pasos_dados": pasos_dados + 1,
            }

        llm = obtener_llm()
        llm_estructurado = llm.with_structured_output(DecisionSupervisor)

        contexto = (
            f"Pregunta original del usuario:\n{_pregunta_original(estado)}\n\n"
            f"Contribuciones hasta ahora:\n{_resumen_contribuciones(estado)}\n\n"
            f"Paso {pasos_dados + 1} de {MAX_PASOS_SUPERVISOR} como máximo."
        )
        decision: DecisionSupervisor = await llm_estructurado.ainvoke(
            [SystemMessage(content=PROMPT_SUPERVISOR), SystemMessage(content=contexto)]
        )

        if decision.siguiente == "FINISH":
            # Acá `instruccion` es la síntesis final para el usuario (no una
            # instrucción para nadie más): eso es lo que tiene que quedar
            # como el mensaje visible, no el log interno de ruteo.
            mensaje = AIMessage(content=decision.instruccion, name="supervisor")
            contribuciones_nuevas: list[Contribucion] = []
        else:
            mensaje = AIMessage(
                content=f"[Supervisor -> {decision.siguiente}] {decision.razon}",
                name="supervisor",
            )
            contribuciones_nuevas = [{"agente": "supervisor", "contenido": decision.instruccion}]

        return {
            "messages": [mensaje],
            "next_agent": decision.siguiente,
            "tarea_completada": decision.siguiente == "FINISH",
            "contribuciones": contribuciones_nuevas,
            "pasos_dados": pasos_dados + 1,
        }

    return nodo_supervisor
