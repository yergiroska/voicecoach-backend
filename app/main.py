"""Punto de entrada de la aplicación FastAPI de VoiceCoach AI."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import dispose_engine, init_engine
from app.routers import health, recordings, users

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Comprueba la configuración al arrancar, sin impedir el arranque.

    La falta de `GROQ_API_KEY` se avisa aquí en vez de hacerla obligatoria en
    `Settings`: un campo requerido lanzaría ValidationError al importar la
    config y el proceso no llegaría a arrancar, dejando también sin servicio a
    /health y /me, que no necesitan Groq para nada. Así el aviso es ruidoso pero
    la degradación queda contenida en POST /recordings (que responde 503).

    Con `DATABASE_URL` se aplica el mismo criterio, pero la degradación es aún
    más suave: sin base de datos POST /recordings sigue transcribiendo y
    devolviendo el texto, simplemente no guarda la fila.
    """
    if not settings.groq_configured:
        logger.warning(
            "GROQ_API_KEY no configurada: POST /recordings responderá 503. "
            "Añádela a tu .env local (ver .env.example). El resto de la API "
            "funciona con normalidad."
        )

    # Se inicializa aquí, al arrancar, y no en la primera petición que necesite
    # la base de datos: `create_async_engine` no abre ninguna conexión (solo
    # construye el pool), así que es seguro incluso con PostgreSQL caído, y
    # hacerlo una sola vez evita que dos peticiones simultáneas creen dos pools.
    if not init_engine():
        logger.warning(
            "DATABASE_URL no configurada: POST /recordings transcribirá con "
            "normalidad pero NO guardará las grabaciones. Añádela a tu .env "
            "local (ver .env.example)."
        )

    yield

    # Cerrar el pool al apagar. Sin esto, con `--reload` las conexiones se
    # acumulan en cada recarga hasta agotar el `max_connections` de PostgreSQL.
    await dispose_engine()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
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
