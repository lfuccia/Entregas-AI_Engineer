# Observabilidad y resiliencia

## Los tres pilares de la observabilidad

Logs, métricas y trazas distribuidas cubren preguntas distintas. Los logs
responden "qué pasó exactamente en este evento puntual" y son ideales para
investigar un caso específico. Las métricas (contadores, gauges, histogramas)
responden "cómo se comporta el sistema en agregado a lo largo del tiempo" —
por ejemplo, p95 de latencia por endpoint— y son baratas de almacenar a
largo plazo comparadas con logs completos. Las trazas distribuidas siguen
una request individual a través de múltiples servicios, mostrando cuánto
tiempo se gastó en cada salto; son las que permiten identificar cuál de los
cinco microservicios involucrados en una request lenta es el responsable
del retraso.

## Circuit breaker

Cuando un servicio dependiente empieza a fallar o a responder muy lento, un
circuit breaker corta las llamadas hacia ese servicio después de detectar
un umbral de errores (por ejemplo, más del 50% de fallos en los últimos 20
requests), devolviendo un error inmediato (o una respuesta de fallback) en
lugar de seguir esperando timeouts. Esto evita que los hilos o conexiones
del servicio que llama queden todos bloqueados esperando a un dependiente
caído, lo cual terminaría propagando la falla hacia arriba (efecto
cascada). Pasado un período de "cooldown", el circuit breaker permite pasar
algunas requests de prueba para verificar si el servicio dependiente se
recuperó, antes de volver a cerrar el circuito por completo.

## Reintentos con backoff

Reintentar automáticamente una request que falló por un error transitorio
(timeout, 503) es razonable, pero reintentar inmediatamente y sin límite
puede agravar una caída ya en curso (el servicio caído recibe aún más
tráfico justo cuando menos lo puede procesar). La práctica estándar es
backoff exponencial con jitter: esperar un tiempo que crece exponencialmente
entre reintento y reintento, con algo de aleatoriedad agregada para evitar
que todos los clientes reintenten exactamente al mismo instante ("efecto
manada"). Además, hay que limitar la cantidad máxima de reintentos y nunca
reintentar automáticamente operaciones no idempotentes sin verificar antes
si la operación original ya se completó.

## SLIs, SLOs y error budget

Un SLI (Service Level Indicator) es una métrica concreta, como "porcentaje
de requests exitosas en los últimos 5 minutos". Un SLO (Service Level
Objective) es el objetivo sobre ese indicador, por ejemplo "99.9% de
requests exitosas por mes". La diferencia entre 100% y el SLO acordado es
el "error budget": cuánta falla el sistema puede permitirse antes de
incumplir el objetivo. Si el error budget de un servicio ya se consumió por
completo en la mitad del mes, la práctica recomendada es priorizar
estabilidad sobre nuevas features hasta que el presupuesto se recupere en
el siguiente período.

## Health checks

Un endpoint `/health` que simplemente responde `200 OK` sin verificar nada
da una falsa sensación de seguridad. Un health check útil distingue entre
"liveness" (¿el proceso sigue vivo y puede responder algo?, usado por el
orquestador para decidir si reiniciar el contenedor) y "readiness" (¿el
servicio está listo para recibir tráfico real, con sus dependencias
críticas —base de datos, caché— disponibles?, usado para decidir si
enrutarle tráfico nuevo). Combinar ambos chequeos en un único endpoint
puede hacer que un orquestador reinicie un contenedor sano solo porque una
dependencia externa está temporalmente lenta.
