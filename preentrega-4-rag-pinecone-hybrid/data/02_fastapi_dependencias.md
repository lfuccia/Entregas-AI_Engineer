# FastAPI: inyección de dependencias

## El sistema `Depends`

El sistema de dependencias de FastAPI permite declarar, como parámetro de
un endpoint, una función que FastAPI ejecuta automáticamente antes del
handler y cuyo resultado se inyecta como argumento. Por ejemplo:

```python
def obtener_sesion_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/pedidos")
async def listar_pedidos(db: Session = Depends(obtener_sesion_db)):
    return db.query(Pedido).all()
```

Esto centraliza la lógica de "cómo se abre y cierra una conexión de base de
datos" en un solo lugar, en vez de repetirla en cada endpoint.

## Dependencias con `yield` para cleanup garantizado

Cuando una dependencia usa `yield` en lugar de `return`, todo el código
después del `yield` se ejecuta como cleanup, incluso si el endpoint lanzó
una excepción. Esto es lo que garantiza que una sesión de base de datos o
un archivo abierto se cierren siempre, sin necesidad de un bloque
`try/finally` manual en cada endpoint que use esa dependencia.

## Dependencias anidadas y cacheo por request

Las dependencias pueden depender de otras dependencias (`Depends` dentro de
la firma de otra función usada con `Depends`), formando un árbol. Dentro de
una misma request, FastAPI cachea el resultado de cada dependencia: si dos
endpoints distintos dependen de la misma función `obtener_usuario_actual`,
esa función se ejecuta una sola vez por request, no dos, salvo que se pase
`use_cache=False` explícitamente en el `Depends`.

## Dependencias a nivel de router o de aplicación

Una dependencia no tiene que declararse endpoint por endpoint: se puede
aplicar a todas las rutas de un `APIRouter` (`APIRouter(dependencies=[Depends(verificar_token)])`)
o a toda la aplicación (`FastAPI(dependencies=[...])`). Esto es útil para
requisitos transversales como autenticación o rate limiting, que deben
aplicarse a decenas de endpoints sin repetir la misma línea en cada uno.

## Dependencias como clases

Cuando una dependencia necesita configuración (por ejemplo, un límite de
paginación distinto por endpoint), se puede implementar como una clase con
`__call__`: `Depends(ValidadorDePaginacion(max_limit=100))`. Esto permite
reutilizar la misma lógica de validación con distintos parámetros en
distintos endpoints, sin duplicar código.
