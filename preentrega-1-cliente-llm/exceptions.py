"""
Taxonomía de errores propia del cliente unificado.

Cada cliente de proveedor (OpenAIClient, AnthropicClient) traduce las
excepciones específicas de su SDK (openai.*, anthropic.*) a esta jerarquía
común, para que AsyncLLMManager no tenga que conocer los detalles de cada
SDK al decidir si vale la pena reintentar.
"""


class LLMClientError(Exception):
    """Error base del cliente unificado."""


class LLMTransientError(LLMClientError):
    """
    Error transitorio: problemas de red, timeout, rate limit, o un 5xx del
    proveedor. Vale la pena reintentar.
    """


class LLMFatalError(LLMClientError):
    """
    Error no transitorio: API key inválida, request mal formado, modelo
    inexistente, etc. Reintentar no cambia el resultado.
    """


class LLMStreamInterrupted(LLMClientError):
    """
    El stream se cortó después de haber emitido al menos un fragmento de
    texto. No se puede "deshacer" el contenido parcial ya entregado, así
    que se señaliza en vez de reintentar la respuesta completa desde cero.
    """
