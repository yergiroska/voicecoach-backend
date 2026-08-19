"""Dependencies de FastAPI reutilizables.

`get_current_user` es la puerta de entrada de autenticación: extrae el ID token
del header `Authorization: Bearer <token>`, lo verifica contra las claves
públicas de Google y devuelve el usuario. Los endpoints que la usen quedan
protegidos; los que no la usen (p. ej. `/health`) siguen siendo públicos.
"""

import logging
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.firebase_auth import (
    FirebaseAuthUnavailableError,
    InvalidFirebaseTokenError,
    verify_firebase_token,
)
from app.models.user import CurrentUser

logger = logging.getLogger(__name__)

# `auto_error=True` hace que FastAPI responda 401 con `WWW-Authenticate: Bearer`
# cuando falta el header o el esquema no es Bearer. Además registra el esquema en
# OpenAPI, así que en /docs aparece el botón "Authorize" para pegar un token.
bearer_scheme = HTTPBearer(
    scheme_name="Firebase ID token",
    bearerFormat="JWT",
    description="ID token de Firebase obtenido en el cliente con `getIdToken()`.",
    auto_error=True,
)

_UNAUTHORIZED_HEADERS = {"WWW-Authenticate": "Bearer"}


# Sync (`def`, no `async def`) a propósito: verificar el token puede hacer una
# petición HTTP a Google cuando la caché de claves está vacía o caducada.
# FastAPI ejecuta las dependencies sync en un threadpool, así que ese bloqueo no
# congela el event loop. Con `async def` sí lo haría.
def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)],
) -> CurrentUser:
    """Devuelve el usuario autenticado a partir del ID token del request.

    Raises:
        HTTPException 401: el token es inválido, está expirado o es de otro proyecto.
        HTTPException 503: no se pudieron obtener las claves públicas de Google.
    """
    try:
        claims = verify_firebase_token(credentials.credentials)
    except InvalidFirebaseTokenError as exc:
        # El motivo exacto va al log (útil en desarrollo) pero no a la respuesta:
        # al cliente le basta saber que debe renovar el token.
        logger.warning("Rechazado un ID token de Firebase: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de Firebase inválido o expirado.",
            headers=_UNAUTHORIZED_HEADERS,
        ) from exc
    except FirebaseAuthUnavailableError as exc:
        # No es culpa del cliente: su token puede ser válido. Devolver 401 aquí
        # provocaría logouts en cascada en el móvil por un fallo de red nuestro.
        logger.error("Verificación de Firebase no disponible: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo verificar la autenticación en este momento.",
        ) from exc

    return CurrentUser.from_claims(claims)


# Atajo para anotar endpoints: `user: CurrentUserDep`.
CurrentUserDep = Annotated[CurrentUser, Depends(get_current_user)]