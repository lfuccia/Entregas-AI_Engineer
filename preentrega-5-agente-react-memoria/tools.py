"""
tools.py
========

Herramientas del agente, definidas con `@tool` de LangChain. Simulan una
base de datos interna de clientes y pedidos (no hay ninguna base de datos
real ni llamada de red: son diccionarios en memoria, a propósito, para que
el proyecto se pueda correr y testear sin infraestructura extra).

Los docstrings son la ÚNICA información que el LLM tiene para decidir qué
herramienta usar y con qué argumentos — por eso son largos y explícitos
(ver "Errores Comunes a Evitar" del enunciado: una descripción vaga es la
causa más común de que el agente no llame a la herramienta esperada).
"""

from __future__ import annotations

import unicodedata

from langchain_core.tools import tool

# -- "Base de datos" simulada -------------------------------------------------

_CLIENTES: dict[int, str] = {
    101: "Laura Fernández",
    102: "Carlos Méndez",
    103: "Martina Gómez",
    104: "Diego Gómez",
    105: "Sofía Ramírez",
}

_PEDIDOS: dict[int, dict] = {
    101: {"pedidos": 5, "total": 32000, "ultimo_pedido": {"id": "P-1980", "fecha": "2026-09-01", "monto": 6100}},
    102: {"pedidos": 3, "total": 14500, "ultimo_pedido": {"id": "P-2044", "fecha": "2026-08-30", "monto": 5200}},
    103: {"pedidos": 7, "total": 58900, "ultimo_pedido": {"id": "P-2091", "fecha": "2026-09-05", "monto": 9200}},
    # 104 (Diego Gómez) existe como cliente pero no tiene pedidos: caso de
    # "información incompleta" para probar el ciclo de retorno.
    105: {"pedidos": 2, "total": 9100, "ultimo_pedido": {"id": "P-1877", "fecha": "2026-08-12", "monto": 4000}},
}


def _sin_tildes(texto: str) -> str:
    normalizado = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in normalizado if not unicodedata.combining(c)).lower().strip()


@tool
def buscar_cliente(nombre_o_id: str) -> dict:
    """Busca un cliente en la base de datos interna de la empresa, por
    nombre (búsqueda parcial, sin distinguir mayúsculas/minúsculas ni
    tildes: "gomez" encuentra "Gómez") o por ID numérico exacto.

    Usá SIEMPRE esta herramienta primero cuando el usuario mencione un
    cliente por su NOMBRE (por ejemplo "Martina Gómez", "el cliente Ramírez")
    en vez de por su ID numérico, para resolver cuál es su `cliente_id`
    antes de poder consultar sus pedidos con `buscar_pedidos`. Si el usuario
    ya te dio el ID numérico directamente (ej. "cliente 102"), no hace falta
    llamar a esta herramienta: pasale ese número directo a `buscar_pedidos`.

    Devuelve un diccionario con alguna de estas formas:
    - Coincidencia única: {"encontrado": true, "cliente_id": <int>, "nombre": <str>}.
    - Nombre ambiguo (varios clientes coinciden, ej. "Gómez" matchea a más
      de uno): {"encontrado": false, "ambiguo": true, "candidatos": [
      {"cliente_id": <int>, "nombre": <str>}, ...]}. En este caso NO
      inventes ni adivines cuál es el correcto: preguntale al usuario cuál
      de los candidatos quiso decir, y volvé a llamar a esta herramienta
      (o directamente a `buscar_pedidos` si ya te da el ID) una vez que
      aclare.
    - Sin coincidencias: {"encontrado": false, "error": "<mensaje>"}. En
      este caso avisale al usuario que no encontraste ese cliente, no
      inventes datos.

    Args:
        nombre_o_id: Nombre completo o parcial del cliente (ej. "Gómez",
            "Martina Gómez"), o su ID numérico como string (ej. "102").
    """
    valor = nombre_o_id.strip()

    if valor.isdigit():
        cliente_id = int(valor)
        nombre = _CLIENTES.get(cliente_id)
        if nombre is None:
            return {"encontrado": False, "error": f"No existe ningún cliente con ID {cliente_id}."}
        return {"encontrado": True, "cliente_id": cliente_id, "nombre": nombre}

    objetivo = _sin_tildes(valor)
    candidatos = [
        {"cliente_id": cid, "nombre": nombre}
        for cid, nombre in _CLIENTES.items()
        if objetivo in _sin_tildes(nombre)
    ]

    if len(candidatos) == 1:
        return {"encontrado": True, **candidatos[0]}
    if len(candidatos) > 1:
        return {"encontrado": False, "ambiguo": True, "candidatos": candidatos}
    return {"encontrado": False, "error": f"No se encontró ningún cliente que coincida con '{nombre_o_id}'."}


@tool
def buscar_pedidos(cliente_id: int) -> dict:
    """Consulta la base de datos interna de pedidos de un cliente, dado su
    ID numérico exacto (no su nombre — si solo tenés el nombre, resolvé
    primero el ID con `buscar_cliente`).

    Devuelve un diccionario con:
    - Si el cliente existe y tiene pedidos: {"cliente_id": <int>,
      "pedidos": <int, cantidad total de pedidos>, "total": <float, suma en
      pesos de todos sus pedidos>, "ultimo_pedido": {"id": <str>, "fecha":
      <str YYYY-MM-DD>, "monto": <float>}}.
    - Si el cliente existe pero todavía no tiene ningún pedido registrado:
      {"cliente_id": <int>, "pedidos": 0, "total": 0, "mensaje": "..."}.
      Esto NO es un error: respondé la pregunta con esta información
      (0 pedidos), no reintentes ni asumas que hubo una falla.
    - Si el `cliente_id` no existe en la base: {"error": "<mensaje>"}. En
      este caso es probable que el ID esté mal o que el usuario haya dado
      un nombre que vos convertiste mal a número: considerá usar
      `buscar_cliente` para confirmar el ID correcto antes de responder.

    Args:
        cliente_id: ID numérico del cliente (ej. 102).
    """
    if cliente_id not in _CLIENTES:
        return {"error": f"No existe ningún cliente con ID {cliente_id}."}

    datos = _PEDIDOS.get(cliente_id)
    if datos is None:
        return {
            "cliente_id": cliente_id,
            "pedidos": 0,
            "total": 0,
            "mensaje": "Este cliente no tiene pedidos registrados todavía.",
        }

    return {"cliente_id": cliente_id, **datos}
