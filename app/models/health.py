"""Schemas Pydantic para el endpoint de salud."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Respuesta del endpoint GET /health."""

    status: str
    service: str
    version: str
