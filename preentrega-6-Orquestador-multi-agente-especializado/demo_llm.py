"""
demo_llm.py
===========

LLMs determinísticos de prueba, usados ÚNICAMENTE por `demo_offline.py`
para generar una traza de ejecución reproducible sin gastar llamadas a
ninguna API real:

- `ChatModelDeterminista`: subclase real de `BaseChatModel` (via
  `FakeMessagesListChatModel`, del propio `langchain_core`) que, en vez de
  llamar a un proveedor, va devolviendo — en orden — los mensajes de un
  guion fijo. Sirve para los sub-agentes ReAct (investigador/analista):
  se necesita que sea un `BaseChatModel` de verdad (no un duck-type
  cualquiera) porque `create_react_agent`/`create_agent` componen el
  modelo con el prompt vía el operador `|` de LangChain (`Runnable`), y
  eso exige un `Runnable` real. `bind_tools` se sobreescribe para devolver
  la misma instancia (ignora qué herramientas le pasan): como el guion ya
  trae los `tool_calls` armados a mano, no hace falta bindear nada de
  verdad — el `ToolNode` interno del agente ReAct ejecuta esas llamadas
  contra las herramientas REALES igual.
- `SupervisorDeterminista`: NO es un `BaseChatModel` (no hace falta: en
  `supervisor.py` se lo invoca directo, sin componerlo con `|`). Implementa
  solo `.with_structured_output(schema)` (devuelve la misma instancia) y
  `.ainvoke(...)`, que entrega -- en orden -- objetos `DecisionSupervisor`
  ya construidos.

El grafo, el `ToolNode`, `tools_condition`-equivalente (acá,
`enrutar_desde_supervisor`) y las herramientas que se ejercitan con estos
LLMs falsos son exactamente los reales.
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage


class ChatModelDeterminista(FakeMessagesListChatModel):
    """`responses` (heredado de `FakeMessagesListChatModel`) es el guion:
    la lista de `AIMessage` a devolver en orden, una por cada llamada."""

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ChatModelDeterminista":
        return self


class SupervisorDeterminista:
    """Guion de decisiones (`DecisionSupervisor`) del Supervisor."""

    def __init__(self, guion: list[Any]) -> None:
        self._guion = list(guion)
        self._indice = 0

    def with_structured_output(self, esquema: Any) -> "SupervisorDeterminista":
        return self

    async def ainvoke(self, mensajes: list[AIMessage]) -> Any:
        if self._indice >= len(self._guion):
            raise RuntimeError(
                "El guion del Supervisor de demo se quedó sin pasos "
                "programados (¿el grafo entró en un bucle o el escenario "
                "necesita un paso más?)."
            )
        siguiente = self._guion[self._indice]
        self._indice += 1
        return siguiente
