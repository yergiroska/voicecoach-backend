"""Endpoints del usuario autenticado."""

from fastapi import APIRouter

from app.core.dependencies import CurrentUserDep
from app.models.user import CurrentUser

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