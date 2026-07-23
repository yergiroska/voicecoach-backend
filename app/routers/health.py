"""Endpoint de salud: confirma que el servidor está vivo."""

from fastapi import APIRouter

from app.core.config import settings
from app.models.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Devuelve el estado del servicio."""
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
    )
