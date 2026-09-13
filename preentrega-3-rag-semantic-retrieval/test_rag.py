"""
test_rag.py
===========

Mini-script de prueba asíncrono para el sistema RAG. Ejecuta dos preguntas:

    1. Una pregunta cuya respuesta SÍ está en los documentos de `./data`.
    2. Una "pregunta trampa" cuya respuesta NO está en los documentos,
       para verificar que el modelo no alucina y responde honestamente
       que no tiene esa información.

La primera ejecución indexa los documentos en ChromaDB (puede tardar unos
segundos: descarga el modelo de embeddings la primera vez); las siguientes
reutilizan el vectorstore ya persistido.

Uso:
    python test_rag.py
"""

from __future__ import annotations

import asyncio
import json

from rag_chain import get_rag_response

PREGUNTA_RESPONDIBLE = (
    "¿Qué algoritmo se recomienda para el rate limiting de una API y por "
    "qué es mejor que una ventana fija?"
)

PREGUNTA_TRAMPA = (
    "¿Qué estrategia de sharding recomienda el documento para escalar "
    "horizontalmente la base de datos?"
)


async def _ejecutar_pregunta(titulo: str, pregunta: str) -> None:
    separador = "=" * 70
    print(f"\n{separador}\n{titulo}\n{separador}")
    print(f"Pregunta: {pregunta}\n")

    resultado = await get_rag_response(pregunta)

    print("Respuesta:")
    print(f"  {resultado.respuesta}\n")
    print(f"Fuentes citadas: {resultado.fuentes or '(ninguna)'}")
    print("\nObjeto validado (RespuestaRAG):")
    print(json.dumps(resultado.model_dump(mode="json"), indent=2, ensure_ascii=False))


async def main() -> None:
    await _ejecutar_pregunta(
        "PRUEBA 1: pregunta respondible (la respuesta está en los documentos)",
        PREGUNTA_RESPONDIBLE,
    )
    await _ejecutar_pregunta(
        "PRUEBA 2: pregunta trampa (la respuesta NO está en los documentos)",
        PREGUNTA_TRAMPA,
    )

    print(
        "\nRevisá arriba: en la Prueba 2, el modelo debería decir "
        "explícitamente que no tiene esa información en los documentos "
        "(y 'fuentes' debería venir vacío o no relacionado a sharding), "
        "en vez de inventar una respuesta plausible."
    )


if __name__ == "__main__":
    asyncio.run(main())
