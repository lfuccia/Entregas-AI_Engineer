# FastAPI: enrutamiento y parámetros

## Path parameters

FastAPI declara parámetros de ruta directamente en la firma de la función,
usando type hints de Python estándar. Por ejemplo, `@app.get("/items/{item_id}")`
junto con `async def leer_item(item_id: int)` hace que FastAPI convierta y
valide automáticamente el segmento de la URL a `int`, devolviendo un error
`422 Unprocessable Entity` con un detalle claro si alguien manda un valor
que no se puede convertir (por ejemplo, `/items/abc`). No hace falta
parsear manualmente strings ni escribir validación a mano: el tipo
declarado en la función ES el contrato.

## Query parameters

Cualquier parámetro de la función que no aparezca en el path de la ruta se
interpreta automáticamente como query parameter. `async def listar(skip: int
= 0, limit: int = 10)` genera un endpoint que acepta `?skip=5&limit=20`,
con valores por default si no se especifican. Parámetros opcionales se
declaran con `Optional[str] = None` (o `str | None = None` en Python
moderno).

## Orden de las rutas y rutas más específicas primero

Un error frecuente es declarar `/items/{item_id}` antes que
`/items/latest`. Como FastAPI evalúa las rutas en el orden en que fueron
declaradas, una request a `/items/latest` puede terminar matcheando
`/items/{item_id}` con `item_id="latest"` (y fallar la validación si
`item_id` es `int`) en lugar de llegar al handler específico. La regla
práctica es declarar siempre las rutas estáticas o más específicas antes
que las rutas con parámetros variables.

## APIRouter para organizar módulos grandes

Cuando una aplicación crece más allá de un puñado de endpoints, conviene
dividir las rutas en múltiples `APIRouter()` (uno por dominio: usuarios,
pedidos, pagos) y incluirlos en la app principal con
`app.include_router(router, prefix="/usuarios", tags=["usuarios"])`. Esto
evita un único archivo `main.py` de miles de líneas y permite versionar o
desactivar módulos completos de forma independiente.

## Respuestas con modelos explícitos

Declarar `response_model=UsuarioOut` en el decorador del endpoint hace que
FastAPI filtre automáticamente los campos de la respuesta según ese
esquema, aunque la función devuelva un objeto con más campos (por ejemplo,
un modelo de base de datos que incluye el hash de la contraseña). Esto
evita filtrar accidentalmente campos sensibles que existen en el objeto
interno pero que nunca deberían salir en la respuesta HTTP.
