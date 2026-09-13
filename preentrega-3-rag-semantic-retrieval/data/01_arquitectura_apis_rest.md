# Arquitectura de APIs REST

## Principios de diseño

Una API REST bien diseñada expone recursos (sustantivos, no verbos) a través
de URLs predecibles: `/usuarios`, `/usuarios/{id}/pedidos`. Los verbos HTTP
(GET, POST, PUT, PATCH, DELETE) expresan la acción; el endpoint nunca debería
llamarse `/crearUsuario` u `/obtenerPedidos`. Esta separación entre recurso y
acción es lo que hace que una API sea predecible para quien la consume por
primera vez.

## Versionado

Cuando una API va a tener múltiples versiones conviviendo (por ejemplo,
mientras se migran clientes de v1 a v2), la práctica recomendada es versionar
en la URL (`/v1/usuarios`) o en un header (`Accept: application/vnd.miapi.v2+json`).
Versionar en la URL es más simple de debuguear y de cachear en un CDN;
versionar por header mantiene las URLs más limpias pero complica el testing
manual. Cualquiera sea la elección, lo importante es no romper contratos:
un campo que un cliente ya consume no debería cambiar de tipo ni desaparecer
sin pasar primero por un período de deprecación anunciado.

## Idempotencia

Los métodos GET, PUT y DELETE deben ser idempotentes: ejecutarlos una vez o
diez veces con los mismos parámetros produce el mismo resultado final en el
servidor. POST, en cambio, típicamente no lo es (cada llamada crea un recurso
nuevo). Para operaciones críticas como pagos, se recomienda aceptar un header
`Idempotency-Key` generado por el cliente: si el servidor recibe la misma key
dos veces, devuelve la respuesta ya calculada en lugar de procesar la
operación de nuevo. Esto evita duplicar cobros cuando el cliente reintenta
una request que tuvo timeout sin saber si el servidor llegó a procesarla.

## Paginación

Ningún endpoint que devuelva una colección debería responder "todos los
registros" sin límite. Las dos estrategias más comunes son paginación por
offset (`?page=2&page_size=50`) y paginación por cursor
(`?cursor=eyJpZCI6MTIzfQ&limit=50`). La paginación por offset es más simple
de implementar pero se degrada con tablas grandes (el `OFFSET` de SQL sigue
escaneando las filas saltadas) y puede duplicar o saltear registros si la
tabla cambia entre páginas. La paginación por cursor —basada en un puntero
opaco al último registro visto— es más eficiente y estable ante escrituras
concurrentes, aunque no permite saltar directamente a "la página 10".

## Códigos de estado

Devolver siempre `200 OK` con un campo `"error": true` en el body es un
antipatrón: rompe el cacheo HTTP estándar y obliga a cada cliente a parsear
el body para saber si algo salió mal. Los códigos de estado existen para
esto: `400` para errores de validación del cliente, `401` sin autenticación,
`403` sin permisos, `404` recurso inexistente, `409` conflicto (por ejemplo,
un email ya registrado), `422` para errores de validación semántica, y `5xx`
reservado exclusivamente para errores del servidor.
