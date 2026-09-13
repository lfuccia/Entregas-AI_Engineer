# Bases de datos y estrategias de caching

## Elegir el motor de base de datos

PostgreSQL es la opción por defecto razonable para la mayoría de los
sistemas transaccionales: soporta JSONB para datos semi-estructurados,
tiene extensiones maduras (PostGIS para geolocalización, pgvector para
búsqueda vectorial) y un optimizador de consultas robusto. MongoDB tiene
sentido cuando el esquema realmente varía documento a documento y las
consultas no requieren joins complejos. Para series temporales de alto
volumen (métricas, logs) conviene una base especializada como TimescaleDB
o ClickHouse en lugar de forzar esos datos en una tabla relacional genérica.

## Índices

Un índice acelera las lecturas pero encarece cada escritura (el motor tiene
que actualizar el índice además de la tabla) y ocupa espacio en disco. La
regla práctica es indexar las columnas que aparecen en cláusulas `WHERE`,
`JOIN` y `ORDER BY` de las consultas más frecuentes, y evitar indexar
columnas con baja cardinalidad (por ejemplo, un booleano) salvo que se use
un índice parcial (`WHERE activo = true`). Un error común es crear un índice
por cada columna "por las dudas": eso ralentiza los `INSERT`/`UPDATE` sin
aportar beneficio real si esas columnas no se consultan.

## Pool de conexiones

Cada conexión a PostgreSQL consume memoria en el servidor (varios MB por
conexión activa) y el número de conexiones concurrentes que soporta un
servidor no es ilimitado. Cuando una aplicación abre una conexión nueva por
cada request en lugar de reutilizar un pool, el servidor de base de datos
puede agotar sus conexiones disponibles bajo carga, generando timeouts en
cascada. La solución estándar es un pool de conexiones (por ejemplo
PgBouncer delante de PostgreSQL, o el pool integrado del driver) con un
tamaño máximo calculado según la capacidad real del servidor, no según la
cantidad de instancias de la aplicación.

## Caché con Redis

Redis se usa típicamente como caché de lectura delante de la base de datos
principal: las consultas costosas o muy frecuentes se guardan en Redis con
un TTL (time-to-live), y solo se recalculan contra la base de datos cuando
el valor cacheado expira o no existe (patrón "cache-aside"). Los puntos
críticos a resolver son: (1) invalidación —cuando el dato subyacente cambia,
hay que borrar o actualizar la clave en Redis, no esperar a que expire sola,
si la consistencia importa—; y (2) el "cache stampede", que ocurre cuando
una clave muy popular expira y miles de requests simultáneas intentan
recalcularla contra la base de datos al mismo tiempo, tumbándola. Este
segundo problema se mitiga con locks distribuidos o recalculando la clave
un poco antes de que expire (patrón "refresh-ahead").

## Sagas y consistencia eventual

En arquitecturas con múltiples servicios y bases de datos separadas, una
transacción que involucra varios servicios no puede envolverse en un solo
`COMMIT` de SQL. El patrón Saga resuelve esto encadenando una serie de
transacciones locales, cada una con su operación de compensación
correspondiente por si un paso posterior falla (por ejemplo, si el pago
falla después de reservar stock, se ejecuta la compensación "liberar
stock"). Esto implica aceptar consistencia eventual: durante una ventana de
tiempo corta, distintas partes del sistema pueden ver el estado
intermedio de la operación.
