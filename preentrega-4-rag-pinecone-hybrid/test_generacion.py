"""
test_generacion.py
===================

Mini-script de prueba END-TO-END: recuperación híbrida (BM25 + Pinecone)
+ generación de respuesta con un LLM gratuito (Groq/Llama). No es parte
del checklist obligatorio de esta pre-entrega (que pide solo ingesta +
recuperador híbrido + evaluate.py), pero sirve como evidencia real y
concreta para pegar en el README: acá no se miden fragmentos recuperados,
se ve la respuesta final en texto.

Requiere:
    - Haber corrido `python ingest.py` al menos una vez (Pinecone poblado).
    - `GROQ_API_KEY` en tu `.env` (gratis en https://console.groq.com/keys).

Uso:
    python test_generacion.py
"""

from __future__ import annotations

from rag_system import RAGSystem

PREGUNTAS_DE_PRUEBA = [
    # Respondible: está en 03_fastapi_validacion_pydantic.md
    "¿Por qué conviene no reutilizar el mismo modelo Pydantic para el input y el output de un endpoint?",
    # Respondible: está en 04_faq_fastapi.json
    "¿Qué servidor ASGI se usa típicamente para correr FastAPI en producción?",
    # Pregunta trampa: NO está en ningún documento del dataset
    "¿Cómo se configura el rate limiting con Redis en una aplicación FastAPI?",
]


def main() -> None:
    rag = RAGSystem(top_k=5)

    for i, pregunta in enumerate(PREGUNTAS_DE_PRUEBA, start=1):
        print("\n" + "=" * 78)
        print(f"[{i}] Pregunta: {pregunta}")
        print("=" * 78)

        resultado = rag.answer(pregunta)

        print(f"\nRespuesta:\n  {resultado.respuesta}")
        print(f"\nFuentes citadas: {resultado.fuentes}")

    print(
        "\nRevisá la pregunta [3]: es una trampa (rate limiting con Redis no "
        "está en ningún documento del dataset). El modelo debería decir "
        "explícitamente que no tiene esa información, no inventarla."
    )


if __name__ == "__main__":
    main()
