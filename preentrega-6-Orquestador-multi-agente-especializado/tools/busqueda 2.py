"""
tools/busqueda.py
==================

Herramienta del Agente de Investigación. Por defecto (`SEARCH_PROVIDER=
simulado`) busca sobre un pequeño corpus propio en memoria — reseñas de
usuarios sobre el lanzamiento de un producto ficticio — para que el
proyecto se pueda correr y testear gratis y sin red. Si se configura
`SEARCH_PROVIDER=tavily` con una `TAVILY_API_KEY` real, se usa búsqueda
web de verdad vía Tavily en su lugar (mismo nombre de herramienta, mismo
contrato de entrada/salida — el agente no necesita cambiar).
"""

from __future__ import annotations

from langchain_core.tools import tool

from config import SEARCH_PROVIDER, TAVILY_API_KEY

# -- Corpus simulado ("fuente externa" de reseñas) ---------------------------

_RESENAS: list[dict] = [
    {
        "id": "r1",
        "fuente": "TiendaApp Reviews",
        "texto": "El nuevo auricular Aurora X2 tiene un sonido excelente y la "
        "batería dura muchísimo más que el modelo anterior. Totalmente "
        "recomendado.",
        "calificacion": 5,
    },
    {
        "id": "r2",
        "fuente": "TiendaApp Reviews",
        "texto": "Buen producto pero la app para configurarlo se cuelga "
        "seguido. El audio en sí es muy bueno.",
        "calificacion": 3,
    },
    {
        "id": "r3",
        "fuente": "Foro TecnoLatam",
        "texto": "Decepcionante. Prometían cancelación de ruido de nivel "
        "premium y en la práctica es mediocre, igual que la generación "
        "pasada.",
        "calificacion": 2,
    },
    {
        "id": "r4",
        "fuente": "TiendaApp Reviews",
        "texto": "Relación precio-calidad increíble. Lo compré para el "
        "gimnasio y no se cae ni se corta la conexión bluetooth.",
        "calificacion": 5,
    },
    {
        "id": "r5",
        "fuente": "Blog GadgetsHoy",
        "texto": "Un lanzamiento sólido, con mejoras reales sobre la versión "
        "anterior, aunque el precio quedó un poco por encima de la "
        "competencia directa.",
        "calificacion": 4,
    },
]


def _buscar_simulado(query: str) -> dict:
    objetivo = query.lower()
    palabras_clave = [p for p in objetivo.split() if len(p) > 3]

    if not palabras_clave:
        coincidencias = _RESENAS
    else:
        coincidencias = [
            r for r in _RESENAS if any(p in r["texto"].lower() for p in palabras_clave)
        ] or _RESENAS  # si ninguna palabra matchea, devolvemos el corpus completo

    return {
        "proveedor": "simulado",
        "consulta": query,
        "resultados": coincidencias,
    }


def _buscar_tavily(query: str) -> dict:
    from langchain_tavily import TavilySearch

    herramienta = TavilySearch(max_results=5, tavily_api_key=TAVILY_API_KEY)
    resultados = herramienta.invoke({"query": query})
    return {"proveedor": "tavily", "consulta": query, "resultados": resultados}


@tool
def buscar_informacion(query: str) -> dict:
    """Busca información externa relevante para una consulta (por defecto,
    sobre un corpus propio de reseñas de usuarios acerca del lanzamiento de
    un producto; si se configuró Tavily, busca en la web real). Usala
    SIEMPRE que necesites datos, opiniones o hechos que no tengas ya en la
    conversación — no inventes reseñas ni cifras.

    Devuelve {"proveedor": "simulado"|"tavily", "consulta": <str>,
    "resultados": [...]}. Cada resultado simulado trae "texto" (la reseña),
    "fuente" y "calificacion" (1 a 5) — pasale ese texto tal cual al Agente
    de Análisis si te piden evaluar sentimiento o métricas, no lo
    resumas vos.

    Args:
        query: qué buscar, en lenguaje natural (ej. "opiniones sobre el
            lanzamiento del auricular Aurora X2").
    """
    if SEARCH_PROVIDER == "tavily" and TAVILY_API_KEY:
        return _buscar_tavily(query)
    return _buscar_simulado(query)
