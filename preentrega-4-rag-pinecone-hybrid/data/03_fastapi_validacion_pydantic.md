# FastAPI: validación de datos con Pydantic

## El body de una request como modelo Pydantic

Cuando un parámetro de un endpoint es una clase que hereda de
`pydantic.BaseModel`, FastAPI entiende automáticamente que ese parámetro
viene del body JSON de la request (no de query ni de path), lo parsea, lo
valida contra el esquema y devuelve `422` con el detalle exacto de qué
campo falló si el body no cumple el esquema. Esto es lo que permite
escribir `async def crear_usuario(usuario: UsuarioIn)` sin ningún código
manual de parseo de JSON ni de validación.

## Modelos de entrada vs modelos de salida

Una práctica recomendada es no reutilizar el mismo modelo Pydantic para lo
que el cliente envía y lo que la API devuelve. Un `UsuarioIn` típicamente
incluye `password: str`; un `UsuarioOut` nunca debería incluir ese campo
(ni siquiera el hash). Definir `UsuarioIn`, `UsuarioOut` y a veces un
`UsuarioEnDB` interno como clases separadas (aunque compartan campos vía
herencia de una clase base común) evita que un campo sensible del modelo
interno termine expuesto por accidente en la respuesta HTTP.

## Validadores personalizados

Además de los tipos básicos, Pydantic permite validadores personalizados
con el decorador `@field_validator` (Pydantic v2) para reglas que no se
expresan solo con el tipo: por ejemplo, que un campo `email` tenga
efectivamente forma de email, o que un campo `fecha_fin` sea posterior a
`fecha_inicio`. Estos validadores se ejecutan automáticamente como parte
del parseo del body, antes de que el código del endpoint reciba el objeto,
así que el handler puede asumir que si el objeto se construyó, ya es
válido.

## `Field` para restricciones y metadatos

`Field(..., min_length=3, max_length=50, description="Nombre visible del usuario")`
permite declarar restricciones (longitud, rango numérico con `gt`/`le`,
regex con `pattern`) directamente en la definición del campo, sin escribir
un validador separado para casos simples. Estos metadatos también alimentan
automáticamente la documentación OpenAPI generada por FastAPI
(`/docs`), así que sirven doblemente como validación y como documentación
para quien consuma la API.

## `model_config` y modelos estrictos

Por default, Pydantic v2 ignora campos extra que no están declarados en el
modelo. Si se quiere rechazar explícitamente cualquier campo no esperado
en el body (por ejemplo, para detectar errores de integración de un
cliente que manda un campo mal escrito), se configura
`model_config = ConfigDict(extra="forbid")` en el modelo, lo que hace que
Pydantic lance un error de validación en vez de ignorar silenciosamente el
campo desconocido.
