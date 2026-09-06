"""
Demo con un modelo real (Anthropic u OpenAI).

Requiere una de estas variables de entorno configuradas:
    ANTHROPIC_API_KEY
    OPENAI_API_KEY

Ejecutar:
    python demo_real.py
"""

from pipeline import build_structured_chain, get_model

TEXTO_CLARO = """
Estamos viendo timeouts intermitentes en el servicio de checkout. El log muestra
'connection pool exhausted' contra la base Postgres, y el API Gateway en AWS
devuelve 504 luego de 30 segundos. El equipo sospecha que el último deploy con
Kubernetes cambió los límites de conexiones del pool de PgBouncer.
"""

TEXTO_AMBIGUO = """
Che, algo se rompió de vuelta y los usuarios se están quejando en el grupo de
soporte. No sé bien qué componente es, pero parece que viene de más temprano.
Alguien que mire porfa.
"""

if __name__ == "__main__":
    model = get_model()
    chain = build_structured_chain(model)

    print("=" * 70)
    print("CASO 1 — Texto técnico claro")
    print("=" * 70)
    resultado_claro = chain.invoke({"texto_entrada": TEXTO_CLARO})
    print(resultado_claro.model_dump_json(indent=2, ensure_ascii=False))

    print()
    print("=" * 70)
    print("CASO 2 — Texto ambiguo (prueba de estrés)")
    print("=" * 70)
    try:
        resultado_ambiguo = chain.invoke({"texto_entrada": TEXTO_AMBIGUO})
        print(resultado_ambiguo.model_dump_json(indent=2, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001 - es un demo, mostramos el error tal cual
        print(f"El pipeline no pudo producir una respuesta válida: {exc}")
