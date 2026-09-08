"""Persistencia de las grabaciones en PostgreSQL.

Esta capa solo escribe filas. No sabe de HTTP y no decide qué hacer cuando algo
falla: traduce cualquier fallo a un único error de dominio,
`RecordingPersistenceError`, y es el router quien elige qué hacer con él. Mismo
reparto de responsabilidades que `audio_storage.py` y `transcription_service.py`.

Un solo tipo de error, y no la lista de excepciones concretas, porque enumerarlas
no funciona: al probarlo con el servidor apagado y con credenciales incorrectas
aparecieron dos excepciones que NO son `SQLAlchemyError` —un
`ConnectionRefusedError` (OSError del sistema) y un `ConnectionDoesNotExistError`
(de asyncpg, que hereda directamente de `Exception`)—. Cada una habría escapado
del `except` del router y devuelto un 500 en el escenario que la persistencia
best-effort existe precisamente para absorber. La lista de excepciones de un
driver es un detalle de implementación que esta capa tiene que absorber, no
publicar.

Esa separación es justo lo que hace posible la regla de `POST /recordings`: la
transcripción ya se hizo y ya se pagó, así que un fallo al guardar la fila no
puede convertirse en un error para el usuario. Quien traduce ese fallo en "201
igual, más un WARNING" es el router; aquí no hay ninguna excepción tragada.

Hay dos puntos de entrada y CADA UNO ABRE SU PROPIA TRANSACCIÓN:
`persist_recording` (usuario + grabación) y `persist_analysis` (el análisis).

No van juntos en una transacción, y eso es deliberado —cambia lo que se había
previsto al diseñar la tabla—. Con una sola transacción, un fallo al insertar el
análisis haría rollback también de la grabación: se perdería el dato
irremplazable (el audio ya se borró del disco) por un problema del dato
regenerable. Y regenerable es exactamente lo que el análisis es: vive en su
propia tabla para poder repetirse, y una grabación sin análisis es un estado
válido y previsto (`analysis: null` en la respuesta).

Así que el aislamiento de fallos sigue el valor del dato, no la comodidad de
una transacción única. El orden importa y no es negociable: primero la
grabación, después el análisis. `analyses.recording_id` es una clave ajena, de
modo que sin la fila de la grabación el análisis no puede existir.
"""

import logging

from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import session_scope
from app.db.models import Analysis, Recording, User
from app.models.analysis import RecordingAnalysis

logger = logging.getLogger(__name__)

# Tope del motivo que se incrusta en los errores de esta capa. Lo que importa
# (clase de la excepción y constraint que saltó) va al principio.
_MAX_MOTIVO_CHARS = 300


def _motivo(exc: BaseException) -> str:
    """Resume una excepción de la base de datos SIN volcar datos del usuario.

    Se queda con la PRIMERA LÍNEA, y no es un detalle cosmético: es lo que
    impide que la voz del usuario acabe en los logs.

    `hide_parameters=True` (ver `app/core/database.py`) tapa los parámetros que
    SQLAlchemy añade al mensaje, pero no tapa el `DETAIL:` que añade
    PostgreSQL, que en una violación de CHECK incluye **la fila completa que
    falló** —con el texto transcrito o el resumen del modelo dentro—. Ese DETAIL
    va en líneas siguientes, así que quedarse con la primera lo elimina y
    conserva justo la parte accionable:

        (asyncpg.IntegrityError) <CheckViolationError>: el nuevo registro para
        la relación «analyses» viola la restricción «check»
        «ck_analyses_clarity_score_range»          <- se conserva
        DETAIL: La fila que falla contiene (..., 'lo que dijo el usuario', ...)
                                                    <- se descarta

    Con el nombre del constraint en el mensaje, un WARNING sigue siendo
    suficiente para diagnosticar sin necesidad del volcado de la fila.
    """
    primera_linea = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    return primera_linea[:_MAX_MOTIVO_CHARS]


class RecordingPersistenceError(Exception):
    """No se pudo guardar la grabación en la base de datos.

    Envuelve cualquier motivo: no hay `DATABASE_URL`, PostgreSQL no responde,
    las credenciales son incorrectas, la migración no se aplicó, un constraint
    saltó... Al router le da igual la causa —en todos los casos la respuesta es
    la misma, 201 más un WARNING—, y tener un solo tipo evita que una excepción
    nueva del driver se cuele sin capturar.
    """


class AnalysisPersistenceError(Exception):
    """No se pudo guardar el análisis en la base de datos.

    Separado de `RecordingPersistenceError` porque las dos cosas se guardan por
    separado y fallan por separado, y porque la gravedad no es la misma: perder
    una grabación es perder la transcripción de una voz que ya no existe;
    perder un análisis es perder algo que se puede volver a calcular sobre esa
    transcripción, que sigue en la base.

    El router los distingue solo para el mensaje del log: en ambos casos la
    respuesta al cliente es 201.
    """


async def ensure_user(session: AsyncSession, uid: str) -> None:
    """Da de alta al usuario si es su primera grabación.

    No hay endpoint de registro: el usuario "existe" para el backend la primera
    vez que sube algo. Por eso el alta es un upsert aquí y no un paso aparte
    —tampoco en `/me`, que sigue sin tocar la base de datos a propósito para
    seguir funcionando cuando PostgreSQL esté caído—.

    `ON CONFLICT DO NOTHING` en lugar de "consultar y luego insertar": esa
    versión tiene una carrera (dos grabaciones simultáneas del mismo usuario
    nuevo verían las dos que no existe) y cuesta una consulta extra siempre.
    Aquí el caso normal —usuario ya conocido— es un único statement que no hace
    nada.

    `created_at` no se pasa: lo pone el `server_default` de PostgreSQL.
    """
    statement = (
        pg_insert(User)
        .values(uid=uid)
        .on_conflict_do_nothing(index_elements=["uid"])
    )
    await session.execute(statement)


async def insert_recording(
    session: AsyncSession,
    *,
    recording_id: str,
    user_uid: str,
    text: str,
    language: str | None,
    duration_seconds: float | None,
    size_bytes: int,
    content_type: str,
    transcription_model: str,
) -> None:
    """Inserta la fila de la grabación ya transcrita.

    Los parámetros son explícitos en vez de recibir los dataclasses
    `SavedRecording` y `Transcription`: así esta capa no depende de las otras
    dos capas de servicio, y el router es el único sitio que conoce a las tres.

    `created_at` lo pone PostgreSQL. Ojo: no coincide exactamente con el
    timestamp que va dentro de `recording_id` —ese se generó al empezar a
    recibir el audio, este al acabar de transcribirlo—; la diferencia es el
    tiempo de subida más el de Whisper.
    """
    session.add(
        Recording(
            id=recording_id,
            user_uid=user_uid,
            text=text,
            language=language,
            duration_seconds=duration_seconds,
            size_bytes=size_bytes,
            content_type=content_type,
            transcription_model=transcription_model,
        )
    )

    # Flush explícito para que un fallo (violación de constraint, conexión
    # caída) salte AQUÍ y no al hacer commit fuera de esta función, donde el
    # traceback ya no diría qué se estaba insertando.
    await session.flush()


async def persist_recording(
    *,
    recording_id: str,
    user_uid: str,
    text: str,
    language: str | None,
    duration_seconds: float | None,
    size_bytes: int,
    content_type: str,
    transcription_model: str,
) -> None:
    """Guarda usuario y grabación en una sola transacción.

    Es el primero de los dos puntos de entrada del router. Todo o nada DENTRO
    de su alcance: si falla el insert de la grabación, el alta del usuario se
    deshace con el rollback. El análisis va aparte, en `persist_analysis`, y
    después (ver el docstring del módulo).

    Raises:
        RecordingPersistenceError: no se pudo guardar, por cualquier motivo.
    """
    try:
        async with session_scope() as session:
            await ensure_user(session, user_uid)
            await insert_recording(
                session,
                recording_id=recording_id,
                user_uid=user_uid,
                text=text,
                language=language,
                duration_seconds=duration_seconds,
                size_bytes=size_bytes,
                content_type=content_type,
                transcription_model=transcription_model,
            )
    except Exception as exc:
        # `except Exception` a conciencia, y es el único sitio del proyecto
        # donde se justifica: aquí no se está ocultando un fallo, se está
        # traduciendo. Ver el docstring del módulo para las dos excepciones de
        # driver que demostraron que enumerarlas no basta.
        #
        # No se traga nada: el motivo viaja dentro de
        # `RecordingPersistenceError` y el router lo registra con su traceback.
        raise RecordingPersistenceError(
            f"No se pudo guardar la grabación {recording_id}: {_motivo(exc)}"
        ) from exc

    logger.debug("Grabación persistida: %s (uid=%s)", recording_id, user_uid)


async def insert_analysis(
    session: AsyncSession,
    *,
    recording_id: str,
    user_uid: str,
    analysis: RecordingAnalysis,
    raw_response: dict[str, Any],
) -> None:
    """Inserta la fila del análisis.

    Recibe `RecordingAnalysis` —un schema de `app/models/`, vocabulario común—
    y no el `AnalysisOutcome` de `analysis_service`, para no depender de otra
    capa de servicio, igual que `insert_recording` no recibe `Transcription`.

    Esta función es el ÚNICO sitio donde se traduce el contrato público a
    columnas, y ahí hay tres renombrados que conviene tener en un solo lugar:

        scores.clarity     -> clarity_score
        scores.confidence  -> confidence_score
        scores.pace        -> pace_score
        analysis.model     -> analysis_model   (el de análisis, no el de Whisper)

    `filler_words` necesita `model_dump()`: la columna es JSONB y guarda una
    lista de diccionarios, no de objetos Pydantic.

    `id` y `created_at` no se pasan: los pone PostgreSQL (`gen_random_uuid()` y
    `now()`).
    """
    session.add(
        Analysis(
            recording_id=recording_id,
            user_uid=user_uid,
            word_count=analysis.metrics.word_count,
            words_per_minute=analysis.metrics.words_per_minute,
            filler_count=analysis.metrics.filler_count,
            filler_words=[muletilla.model_dump() for muletilla in analysis.metrics.filler_words],
            fillers_analyzed=analysis.metrics.fillers_analyzed,
            clarity_score=analysis.scores.clarity,
            confidence_score=analysis.scores.confidence,
            pace_score=analysis.scores.pace,
            summary=analysis.summary,
            suggestions=list(analysis.suggestions),
            raw_response=raw_response,
            analysis_model=analysis.model,
            prompt_version=analysis.prompt_version,
        )
    )

    # Mismo motivo que en `insert_recording`: que un constraint que salte lo
    # haga aquí, donde el traceback dice qué se estaba insertando, y no al
    # cerrar la transacción fuera de esta función.
    await session.flush()


async def persist_analysis(
    *,
    recording_id: str,
    user_uid: str,
    analysis: RecordingAnalysis,
    raw_response: dict[str, Any],
) -> None:
    """Guarda el análisis en su propia transacción.

    Segundo punto de entrada del router, y hay que llamarlo DESPUÉS de que
    `persist_recording` haya terminado bien. Si la grabación no se guardó, el
    router debe saltarse esta llamada: `analyses.recording_id` es una clave
    ajena y el insert fallaría siempre. El fallo estaría contenido igual (se
    traduce a `AnalysisPersistenceError` y la respuesta sigue siendo 201), pero
    dejaría en el log un WARNING que culpa al análisis de un problema que era
    de la grabación.

    Raises:
        AnalysisPersistenceError: no se pudo guardar, por cualquier motivo.
    """
    try:
        async with session_scope() as session:
            await insert_analysis(
                session,
                recording_id=recording_id,
                user_uid=user_uid,
                analysis=analysis,
                raw_response=raw_response,
            )
    except Exception as exc:
        # `except Exception` por el mismo motivo que en `persist_recording`:
        # traducir, no ocultar. Las excepciones que puede lanzar asyncpg no se
        # pueden enumerar de forma fiable (ver el docstring del módulo).
        raise AnalysisPersistenceError(
            f"No se pudo guardar el análisis de {recording_id}: {_motivo(exc)}"
        ) from exc

    logger.debug(
        "Análisis persistido: %s (uid=%s, prompt=%s)",
        recording_id,
        user_uid,
        analysis.prompt_version,
    )
