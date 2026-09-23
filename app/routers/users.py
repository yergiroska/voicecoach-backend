"""Endpoints del usuario autenticado."""

import logging

from fastapi import APIRouter, HTTPException, status

from app.core.database import SessionDep
from app.core.dependencies import CurrentUserDep
from app.models.baseline import BaselineResponse
from app.models.user import CurrentUser
from app.services.baseline_service import BaselineUnavailableError, get_baseline

logger = logging.getLogger(__name__)

router = APIRouter(tags=["users"])


@router.get(
    "/me",
    response_model=CurrentUser,
    summary="Datos del usuario autenticado",
    responses={
        401: {"description": "Falta el header Authorization, o el token es inválido/expirado."},
        503: {"description": "No se pudieron obtener las claves públicas de Google."},
    },
)
def read_current_user(user: CurrentUserDep) -> CurrentUser:
    """Devuelve el usuario dueño del ID token enviado en `Authorization: Bearer`.

    Sirve al cliente para confirmar que su token es válido y que el backend lo
    reconoce, sin necesidad de tocar la base de datos.
    """
    return user


@router.get(
    "/me/baseline",
    response_model=BaselineResponse,
    summary="Línea base personal y evolución reciente",
    responses={
        401: {"description": "Falta el header Authorization, o el token es inválido/expirado."},
        503: {
            "description": (
                "No se pudieron obtener las claves públicas de Google, o la base "
                "de datos no está disponible."
            )
        },
    },
)
async def read_baseline(
    user: CurrentUserDep,
    # Después de `user` a propósito, como en `POST /recordings`: un request sin
    # token se rechaza con 401 antes de pedir nada a la base de datos.
    session: SessionDep,
) -> BaselineResponse:
    """Compara las sesiones recientes del usuario con su línea base.

    La línea base son sus primeras sesiones válidas, y se compara contra ella la
    media de las más recientes. Las métricas deterministas llevan cifras; los
    scores del modelo, solo una tendencia en palabras (ver `BaselineResponse`).

    Al contrario que `GET /me`, este endpoint SÍ necesita la base de datos, y
    sin ella responde 503: no hay nada que calcular sin el histórico. Un usuario
    sin ninguna grabación no es un error: recibe `status: collecting` con cero
    sesiones.
    """
    try:
        return await get_baseline(session, user.uid)
    except BaselineUnavailableError as exc:
        # ERROR y no WARNING: al revés que en `POST /recordings`, aquí no hay
        # degradación posible, el cliente se queda sin respuesta.
        logger.error("Línea base no disponible (uid=%s): %s", user.uid, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="La base de datos no está disponible en este momento.",
        ) from exc
