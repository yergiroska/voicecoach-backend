"""Punto de entrada de la aplicación FastAPI de VoiceCoach AI."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import health, recordings, users

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

# CORS — desarrollo local: aceptar peticiones desde cualquier origen.
# TODO: restringir allow_origins a los dominios reales antes de producción.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(users.router)
app.include_router(recordings.router)
