# Seguridad y autenticación en APIs

## OAuth2 y OpenID Connect

OAuth2 es un protocolo de autorización, no de autenticación: define cómo una
aplicación obtiene un token para actuar en nombre de un usuario sin conocer
su contraseña. OpenID Connect se construye encima de OAuth2 para agregar
autenticación (identidad verificada del usuario) mediante un ID Token en
formato JWT. El flujo más recomendado para aplicaciones web modernas es
"Authorization Code with PKCE": el cliente nunca maneja el token directamente
en el navegador sin protección, y el intercambio del código por el token
requiere un `code_verifier` que solo conoce el cliente original, mitigando
ataques de interceptación del código de autorización.

## JWT: qué guardar y qué no

Un JWT (JSON Web Token) tiene tres partes: header, payload y firma. El
payload es texto en Base64, **no está encriptado**, así que cualquiera que
tenga el token puede leer su contenido; solo la firma garantiza que no fue
modificado. Por eso nunca se debe guardar información sensible (contraseñas,
números de tarjeta, datos de salud) dentro del payload de un JWT. La
expiración (`exp`) debe ser corta para tokens de acceso (minutos u horas);
para sesiones largas se usa un refresh token de vida más extensa, almacenado
de forma segura (cookie httpOnly, no en `localStorage`, para reducir el
riesgo de robo vía XSS).

## Rate limiting

Limitar la cantidad de requests que un cliente puede hacer en una ventana de
tiempo protege al backend de abuso (intencional o accidental) y de ataques
de fuerza bruta contra endpoints de login. El algoritmo de "token bucket" es
el más usado: cada cliente tiene un balde con capacidad fija de tokens que
se recargan a una tasa constante; cada request consume un token, y si el
balde está vacío la request se rechaza con `429 Too Many Requests` y un
header `Retry-After`. A diferencia de una ventana fija, el token bucket
permite ráfagas cortas de tráfico legítimo sin penalizar al cliente el resto
de la ventana.

## Gestión de secretos

Las credenciales (API keys, contraseñas de base de datos, claves de
firmado) nunca deben vivir en el código fuente ni en variables de entorno
compartidas en texto plano dentro de un repositorio. La práctica recomendada
es un gestor de secretos dedicado (Vault, AWS Secrets Manager, GCP Secret
Manager) que permite rotar credenciales sin redeployar la aplicación, y
llevar un registro de auditoría de quién accedió a qué secreto y cuándo. Un
archivo `.env` es aceptable para desarrollo local, siempre que esté excluido
del control de versiones mediante `.gitignore`.

## Principio de menor privilegio

Cada servicio, usuario o API key debe tener exactamente los permisos que
necesita para cumplir su función, ni uno más. Un error frecuente es que un
servicio de solo lectura de reportes tenga credenciales con permisos de
escritura sobre toda la base de datos "por comodidad". Si esas credenciales
se filtran, el radio de daño posible es mucho mayor de lo necesario. Aplicar
este principio también facilita auditar qué componente pudo haber causado
un incidente, porque los permisos ya delimitan qué acciones son posibles
desde cada origen.
