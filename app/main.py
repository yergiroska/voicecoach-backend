"""Punto de entrada de la aplicación FastAPI de VoiceCoach AI."""

from fastapi import FastAPI

from app.core.config import settings
from app.routers import health

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

app.include_router(health.router)
