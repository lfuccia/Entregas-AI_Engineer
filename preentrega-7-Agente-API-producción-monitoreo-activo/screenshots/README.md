# screenshots/

Acá van las capturas de pantalla que pide el enunciado como evidencia:

1. **Trazas en el dashboard de observabilidad** (Phoenix o LangSmith),
   mostrando al menos una ejecución completa del grafo (supervisor,
   investigador, analista, el nodo de aprobación humana).
2. **Costo por ejecución y latencia p95** que muestra EL DASHBOARD (no el
   número que imprime `scripts/load_test.py` en la terminal — ese es un
   dato aparte, complementario) para la corrida de 5 pedidos
   concurrentes.
3. **Api manager de Fast API** que muestra EL DASHBOARD de la API de producción (pre-entrega 7)
   que expone el sistema multi-agente Supervisor/Investigador/Analista de la pre-entrega 6, 
   con jobs asíncronos, persistencia en Redis, observabilidad activa y un punto de aprobación humana (HITL).
   http://localhost:8000/docs#/

Ver la sección "Observabilidad y prueba de carga" del README principal
para el paso a paso de cómo generarlas.