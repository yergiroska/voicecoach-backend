"""Entorno de Alembic para VoiceCoach AI.

Personalizado respecto a la plantilla `async` de `alembic init` en tres cosas:

1. La URL de la base de datos sale de `app.core.config.settings`, no de
   `alembic.ini`. Así hay una sola fuente de verdad (el `.env`, gitignorado) y
   la contraseña nunca acaba en un archivo versionado.
2. `target_metadata` apunta al `MetaData` de nuestra `Base`, que ya trae la
   `naming_convention`, para que `alembic revision --autogenerate` funcione y
   genere nombres de constraint predecibles.
3. El engine se construye a mano con `create_async_engine` en vez de con
   `async_engine_from_config`. Ese helper lee la URL de `alembic.ini` pasando
   por configparser, que interpreta `%` como sintaxis de interpolación: una
   contraseña con un `%` rompería la conexión con un error incomprensible.
   Saltándose el .ini, la cadena llega intacta.

A diferencia de la app —que arranca sin `DATABASE_URL` y solo avisa con un
WARNING—, aquí la falta de URL es un error fatal: una migración sin base de
datos no tiene ningún sentido, y fallar en silencio dejaría creer que el schema
se aplicó.
"""

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context
from app.core.config import settings
from app.db.models import Base

# Objeto de configuración de Alembic (lee alembic.ini).
config = context.config

# Configura el logging de Python a partir del .ini.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Metadata contra la que se comparan las migraciones en `--autogenerate`.
target_metadata = Base.metadata


def _database_dsn() -> str:
    """La URL de la base de datos, o un error claro si no está configurada."""
    if not settings.database_configured:
        raise RuntimeError(
            "DATABASE_URL no está configurada. Alembic necesita una base de "
            "datos a la que conectarse: añádela a tu `.env` (ver .env.example)."
        )
    return settings.database_dsn


def run_migrations_offline() -> None:
    """Genera el SQL de las migraciones sin conectarse a la base de datos.

    Se usa con `alembic upgrade head --sql`, útil para revisar el DDL antes de
    aplicarlo en producción (Supabase, Fase 9).
    """
    context.configure(
        url=_database_dsn(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # Detecta también cambios de TIPO de columna, no solo columnas
        # añadidas o borradas. Sin esto, cambiar un Integer a BigInteger pasa
        # desapercibido en el autogenerate.
        compare_type=True,
        # Ídem para cambios en los server_default.
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Conecta con la base de datos y aplica las migraciones."""
    # NullPool: es un proceso de un solo uso que se muere al terminar, así que
    # mantener un pool de conexiones no aporta nada.
    connectable = create_async_engine(_database_dsn(), poolclass=pool.NullPool)

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
