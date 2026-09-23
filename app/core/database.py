"""Conexión a PostgreSQL: engine asíncrono y sesiones de SQLAlchemy.

Este módulo solo se ocupa de *cómo* se habla con la base de datos. El schema
vive en `app/db/models.py` y las consultas concretas en `app/services/`.

Dos formas de obtener una sesión, y la diferencia entre ambas importa:

- `session_scope()`: context manager explícito. Lo usa `POST /recordings`, donde
  guardar la fila es *best-effort*: la transcripción ya se hizo (y ya se pagó),
  así que un fallo de base de datos no debe convertirse en un error para el
  usuario. Al ser explícito, la sesión se abre DENTRO del endpoint y su fallo se
  puede capturar ahí mismo.

- `SessionDep`: dependency de FastAPI, para endpoints que SÍ necesitan la base
  de datos para poder responder (leer el histórico, por ejemplo) y que deben
  devolver 503 si no está disponible.

Un endpoint best-effort NO puede usar la dependency: FastAPI resuelve las
dependencies ANTES de ejecutar el cuerpo del endpoint, así que una base de datos
caída abortaría la petición antes incluso de transcribir el audio, que es
exactamente lo contrario de lo que queremos.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from asyncpg.exceptions import PostgresConnectionError
from fastapi import Depends, HTTPException, status
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings

logger = logging.getLogger(__name__)


class DatabaseUnavailableError(Exception):
    """No hay base de datos configurada en el servidor.

    Nunca es culpa del cliente. Es el equivalente de
    `TranscriptionUnavailableError` en la capa de Groq: el router decide si eso
    se traduce a un 503 o si se traga (best-effort).
    """


# Las excepciones que significan "no hay conexión con la base de datos", para
# quien necesite distinguir indisponibilidad (503) de un bug nuestro (500).
# Vive aquí porque este es el módulo que conoce el driver: el resto de capas
# importan la tupla en vez de saber qué lanza asyncpg.
#
# Dos de ellas NO las envuelve SQLAlchemy, y se descubrieron probando contra
# PostgreSQL local, no leyendo documentación:
#
# - `OSError`: puerto cerrado (`ConnectionRefusedError`) o host que no resuelve
#   (`socket.gaierror`). No llegó a haber conexión DBAPI que envolver.
#   `session_scope` ya la traduce a `DatabaseUnavailableError`, pero se incluye
#   por si salta fuera de él.
# - `PostgresConnectionError` de asyncpg: con contraseña incorrecta y con una
#   base inexistente llega un `ConnectionDoesNotExistError`, que hereda de
#   `Exception` y de nada de SQLAlchemy. Sin esta entrada, `get_session` daba
#   500 en esos dos casos (reproducido con un endpoint de prueba).
#
# Ojo: la conexión es PEREZOSA. Se abre en el primer `execute`, no al crear la
# sesión, así que estos errores saltan dentro del cuerpo del endpoint, no al
# resolver la dependency.
CONNECTIVITY_ERRORS: tuple[type[BaseException], ...] = (
    DatabaseUnavailableError,
    OSError,
    PostgresConnectionError,
    OperationalError,
    InterfaceError,
)


# Engine y factoría de sesiones a nivel de módulo: el pool de conexiones debe
# ser único para todo el proceso. Deliberadamente NO se crean en el import, por
# el mismo motivo que el cliente de Groq: así la app arranca sin DATABASE_URL.
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def init_engine() -> bool:
    """Crea el engine y la factoría de sesiones si hay DATABASE_URL.

    Idempotente. La llama el `lifespan` de `app/main.py` al arrancar, y ese
    llamada temprana es lo que evita la carrera de que dos peticiones
    simultáneas creen dos engines (y por tanto dos pools, uno de ellos
    huérfano). `_get_session_factory` la reintenta por si acaso, para que un
    script que importe esto sin pasar por el `lifespan` también funcione.

    `create_async_engine` NO abre ninguna conexión: solo construye el pool. Por
    eso es seguro llamar a esto al arrancar aunque PostgreSQL esté caído — el
    fallo aparecerá en la primera consulta real, no aquí.

    Returns:
        True si quedó inicializado, False si no hay DATABASE_URL configurada.
    """
    global _engine, _session_factory

    if _session_factory is not None:
        return True

    if not settings.database_configured:
        return False

    _engine = create_async_engine(
        settings.database_dsn,
        echo=settings.database_echo,
        # Valida la conexión con un ping antes de entregarla. Cuesta un
        # round-trip, pero evita el clásico "server closed the connection
        # unexpectedly" cuando una conexión del pool lleva rato inactiva; con
        # una DB gestionada (Supabase en la Fase 9, que corta las conexiones
        # ociosas) esto pasa de deseable a necesario.
        pool_pre_ping=True,
        # NO es un ajuste cosmético: es lo que impide que la transcripción
        # acabe en los logs.
        #
        # Por defecto, el `str()` de un error de SQLAlchemy incluye la
        # sentencia Y sus parámetros: `[parameters: {'text': '...'}]`. Como el
        # texto transcrito es uno de esos parámetros, cualquier INSERT fallido
        # que se registre volcaría al log las palabras del usuario. Con esto,
        # SQLAlchemy los sustituye por un aviso.
        #
        # Es la contrapartida técnica de la decisión de no meter el texto en el
        # WARNING de persistencia fallida (ver `app/routers/recordings.py`):
        # sin esta línea, esa decisión se la saltaría el propio mensaje de
        # error. El precio es que al depurar no se ven los valores enviados;
        # `database_echo` sigue mostrando las sentencias.
        hide_parameters=True,
    )
    _session_factory = async_sessionmaker(
        _engine,
        # Sin esto, leer un atributo de un objeto después del commit dispara un
        # refresh contra la DB; en código async eso lanza MissingGreenlet. Con
        # `False` el objeto sigue usable tras el commit.
        expire_on_commit=False,
    )
    logger.info("Engine de base de datos inicializado.")
    return True


async def dispose_engine() -> None:
    """Cierra el pool de conexiones. La llama el `lifespan` al apagar.

    Sin esto, al parar uvicorn las conexiones quedan abiertas hasta que
    PostgreSQL las expira por su cuenta, y con `--reload` se acumulan en cada
    recarga hasta agotar `max_connections`.
    """
    global _engine, _session_factory

    if _engine is None:
        return

    await _engine.dispose()
    _engine = None
    _session_factory = None
    logger.info("Pool de conexiones de base de datos cerrado.")


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Devuelve la factoría de sesiones, inicializándola si hace falta.

    Raises:
        DatabaseUnavailableError: no hay DATABASE_URL configurada.
    """
    if not init_engine():
        raise DatabaseUnavailableError("DATABASE_URL no configurada en el servidor.")

    # `init_engine` devolvió True, así que la factoría existe.
    assert _session_factory is not None
    return _session_factory


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Sesión con transacción: commit al salir, rollback si hubo excepción.

    Uso:
        async with session_scope() as session:
            session.add(...)

    Raises:
        DatabaseUnavailableError: no hay DATABASE_URL configurada, o no se pudo
            establecer la conexión con PostgreSQL.
        SQLAlchemyError: la conexión funcionó pero la operación falló (violación
            de constraint, error de SQL...). Quien llama decide qué hacer con él
            (en `POST /recordings`, tragárselo).
    """
    factory = _get_session_factory()

    try:
        # `factory.begin()` abre la sesión, arranca la transacción, hace commit
        # al salir sin error (rollback si lo hubo) y cierra la sesión. Hacerlo a
        # mano es la forma habitual de olvidarse del `close()`.
        async with factory.begin() as session:
            yield session
    except OSError as exc:
        # IMPORTANTE, y verificado a mano con el servidor apagado: cuando no se
        # puede ni abrir el socket, asyncpg lanza el OSError del sistema
        # (ConnectionRefusedError, socket.gaierror si el host no resuelve...) y
        # SQLAlchemy NO lo envuelve en un SQLAlchemyError, porque no llegó a
        # haber una conexión DBAPI que envolver.
        #
        # Sin esta traducción el error se escaparía por encima de quien captura
        # `SQLAlchemyError` y el endpoint devolvería un 500, justo en el caso
        # más probable —PostgreSQL caído— que la persistencia best-effort
        # existe para absorber.
        #
        # Se traduce aquí, en la capa que conoce el driver, siguiendo el mismo
        # criterio que `transcription_service.py` con los errores de Groq: el
        # router no debería tener que saber qué excepciones lanza asyncpg.
        raise DatabaseUnavailableError(
            f"No se pudo conectar con PostgreSQL: {exc}"
        ) from exc


async def get_session() -> AsyncIterator[AsyncSession]:
    """Dependency de FastAPI para endpoints que requieren base de datos.

    La usa `GET /me/baseline`, el primer endpoint de lectura. NO la usa
    `POST /recordings`, que persiste en modo best-effort con `session_scope()`
    (ver el docstring del módulo).

    Raises:
        HTTPException 503: no hay base de datos configurada o no se pudo conectar.
    """
    # Se apoya en `session_scope`, no en la factoría directamente, para heredar
    # su traducción de los OSError del driver (ver allí).
    try:
        async with session_scope() as session:
            yield session
    except CONNECTIVITY_ERRORS as exc:
        # Solo la indisponibilidad y los fallos de CONECTIVIDAD se traducen a
        # 503. Un IntegrityError o un error de SQL es un bug nuestro y debe
        # salir como 500, no disfrazarse de indisponibilidad temporal.
        logger.error("Base de datos no disponible: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="La base de datos no está disponible en este momento.",
        ) from exc


# Atajo para anotar endpoints: `session: SessionDep`.
SessionDep = Annotated[AsyncSession, Depends(get_session)]
