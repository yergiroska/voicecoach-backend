"""Verificación de ID tokens de Firebase Authentication sin el SDK Admin.

El SDK Admin de Firebase exige una clave privada de cuenta de servicio, que la
política de la organización de Google Cloud no nos permite emitir. En su lugar
se usa `google-auth`, que valida la firma del JWT contra las claves públicas de
Google y solo necesita el Project ID (dato público).

Dos cosas que `google.oauth2.id_token.verify_firebase_token()` NO hace y que
este módulo añade:

1. **No valida el claim `iss`.** A diferencia de `verify_oauth2_token()`, la
   variante de Firebase delega en `verify_token()` sin comprobar el issuer. La
   documentación de Firebase exige `iss == https://securetoken.google.com/<pid>`
   y un `sub` no vacío, así que se comprueban a mano aquí.
2. **No cachea las claves públicas.** Su `_fetch_certs()` hace un GET a Google
   en cada llamada; como esto se ejecuta en una dependency por cada request
   autenticado, se envuelve el transporte en una caché que respeta el
   `Cache-Control: max-age` de la respuesta.

Referencias:
- https://firebase.google.com/docs/auth/admin/verify-id-tokens
- https://google-auth.readthedocs.io/en/master/reference/google.oauth2.id_token.html
"""

import re
import threading
import time
from typing import Any, Mapping

from google.auth import exceptions as google_exceptions
from google.auth import transport
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token

from app.core.config import settings

# Los ID tokens de Firebase llevan `iss` = este prefijo + el Project ID.
_ISSUER_PREFIX = "https://securetoken.google.com/"

_MAX_AGE_RE = re.compile(r"max-age\s*=\s*(\d+)", re.IGNORECASE)


class InvalidFirebaseTokenError(Exception):
    """El token es inválido: firma incorrecta, expirado, u otro proyecto.

    Es culpa del cliente — se traduce a un 401.
    """


class FirebaseAuthUnavailableError(Exception):
    """No se pudieron obtener las claves públicas de Google.

    No es culpa del cliente (su token puede ser perfectamente válido) — se
    traduce a un 503, nunca a un 401.
    """


class _CachedResponse(transport.Response):
    """Copia inmutable en memoria de una respuesta HTTP, reutilizable N veces.

    Necesaria porque el cuerpo de la respuesta original solo se puede consumir
    de forma fiable una vez.
    """

    def __init__(self, status: int, headers: Mapping[str, str], data: bytes):
        self._status = status
        self._headers = headers
        self._data = data

    @property
    def status(self) -> int:
        return self._status

    @property
    def headers(self) -> Mapping[str, str]:
        return self._headers

    @property
    def data(self) -> bytes:
        return self._data


class _CachingRequest:
    """Transporte HTTP para google-auth que cachea los GET correctos.

    Solo se usa para el endpoint de certificados de Google, cuyo contenido rota
    cada pocas horas y viene anunciado por `Cache-Control: max-age`. Las
    respuestas de error no se cachean, para que un fallo puntual no deje la
    autenticación caída durante toda la TTL.
    """

    def __init__(self, fallback_ttl_seconds: int):
        self._request = google_requests.Request()
        self._fallback_ttl = fallback_ttl_seconds
        self._lock = threading.Lock()
        # url -> (instante de caducidad en reloj monotónico, respuesta)
        self._cache: dict[str, tuple[float, _CachedResponse]] = {}

    def __call__(
        self,
        url: str,
        method: str = "GET",
        body: Any = None,
        headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> transport.Response:
        if method != "GET" or body is not None:
            return self._request(url, method=method, body=body, headers=headers, **kwargs)

        now = time.monotonic()
        with self._lock:
            entry = self._cache.get(url)
            if entry is not None and entry[0] > now:
                return entry[1]

        response = self._request(url, method=method, body=body, headers=headers, **kwargs)
        cached = _CachedResponse(response.status, response.headers, response.data)

        if cached.status == 200:
            ttl = _parse_max_age(cached.headers) or self._fallback_ttl
            with self._lock:
                self._cache[url] = (time.monotonic() + ttl, cached)

        return cached


def _parse_max_age(headers: Mapping[str, str]) -> int | None:
    """Extrae el `max-age` (segundos) de un `Cache-Control`, o None si no hay."""
    cache_control = headers.get("cache-control") or headers.get("Cache-Control")
    if not cache_control:
        return None
    match = _MAX_AGE_RE.search(cache_control)
    if match is None:
        return None
    seconds = int(match.group(1))
    return seconds if seconds > 0 else None


# Compartido por todo el proceso: la caché de certificados solo sirve de algo si
# es única, y `requests.Session` (dentro de google_requests.Request) es thread-safe
# para este uso.
_transport_request = _CachingRequest(settings.firebase_certs_cache_ttl_seconds)


def verify_firebase_token(token: str) -> dict[str, Any]:
    """Verifica un ID token de Firebase y devuelve sus claims decodificados.

    Comprueba la firma contra las claves públicas de Google, que `aud` sea
    nuestro Project ID, que `exp`/`iat` sean coherentes, que `iss` sea el
    issuer de Firebase de este proyecto y que `sub` (el uid) no esté vacío.

    Args:
        token: el JWT en crudo, sin el prefijo "Bearer ".

    Returns:
        Los claims del token: `sub` (uid), `email`, `email_verified`,
        `name`, `picture`, `firebase.sign_in_provider`, etc.

    Raises:
        InvalidFirebaseTokenError: token mal formado, firma inválida, expirado
            o emitido para otro proyecto Firebase.
        FirebaseAuthUnavailableError: no se pudieron descargar las claves
            públicas de Google.
    """
    if not token:
        raise InvalidFirebaseTokenError("El token está vacío.")

    project_id = settings.firebase_project_id

    try:
        claims = google_id_token.verify_firebase_token(
            token,
            _transport_request,
            audience=project_id,
            clock_skew_in_seconds=settings.firebase_clock_skew_seconds,
        )
    except google_exceptions.TransportError as exc:
        # Fallo de red al bajar los certificados: el token no tiene la culpa.
        raise FirebaseAuthUnavailableError(
            f"No se pudieron obtener las claves públicas de Google: {exc}"
        ) from exc
    except (ValueError, google_exceptions.GoogleAuthError) as exc:
        # google-auth usa ValueError para firma inválida, token expirado y
        # audience incorrecto.
        raise InvalidFirebaseTokenError(f"Token de Firebase inválido: {exc}") from exc

    # `verify_firebase_token` no valida el issuer: lo hacemos aquí.
    expected_issuer = f"{_ISSUER_PREFIX}{project_id}"
    issuer = claims.get("iss")
    if issuer != expected_issuer:
        raise InvalidFirebaseTokenError(
            f"Issuer inesperado: se esperaba {expected_issuer!r}, se recibió {issuer!r}."
        )

    # `sub` es el uid; Firebase garantiza que es un string no vacío.
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject:
        raise InvalidFirebaseTokenError("El token no trae un `sub` (uid) válido.")

    return dict(claims)