"""Endpoints de grabaciones de audio.

`POST /recordings` recibe el archivo del móvil, lo guarda en disco, lo
transcribe con Groq Whisper de forma síncrona —el cliente espera la
transcripción en la misma respuesta— y guarda la fila en PostgreSQL.

El audio en disco es temporal: se borra cuando la grabación queda persistida.
Se conserva en los dos casos en los que hace falta para recuperar el trabajo:
si falla la transcripción y si falla el guardado (ver `upload_recording`).

Guardar en base de datos es *best-effort* y NO puede hacer fallar la petición:
cuando se llega a ese punto ya se transcribió el audio (y ya se pagó la llamada
a Groq), así que un problema de infraestructura nuestro no debe traducirse en un
error para el usuario. Un fallo de persistencia deja un WARNING para
intervención manual y la respuesta sigue siendo 201.

Después de guardar la grabación se analiza la comunicación (muletillas, ritmo,
claridad, confianza) y se guarda en la tabla `analyses`. Ese tramo es
best-effort de principio a fin y en TRES pasos independientes —calcular las
métricas, preguntar al modelo, guardar la fila—: cualquiera puede fallar sin
que la respuesta deje de ser 201. En el peor caso el cliente recibe la
transcripción con `analysis: null`.

El orden importa: el análisis va DESPUÉS de persistir la grabación, no antes.
La transcripción es el dato que no se puede recuperar (el audio ya se borró),
así que se pone a salvo primero; el análisis se puede repetir sobre el texto
guardado, y para eso existe la tabla aparte.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.core.config import settings
from app.core.dependencies import CurrentUserDep
from app.models.recording import RecordingUploadResponse
from app.services.analysis_service import AnalysisError, AnalysisOutcome, analyze
from app.services.audio_storage import (
    ALLOWED_AUDIO_TYPES,
    AudioTooLargeError,
    UnsupportedAudioTypeError,
    save_recording,
)
from app.services.recording_repository import (
    AnalysisPersistenceError,
    RecordingPersistenceError,
    persist_analysis,
    persist_recording,
)
from app.services.speech_metrics import compute_metrics
from app.services.transcription_service import (
    TranscriptionFailedError,
    TranscriptionUnavailableError,
    transcribe,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["recordings"])

# Margen sobre el límite al comparar contra `Content-Length`: el body multipart
# pesa algo más que el archivo (boundary, Content-Disposition, etc.) y un audio
# justo en el límite no debe rechazarse por esos bytes de sobre.
_MULTIPART_OVERHEAD_ALLOWANCE = 8 * 1024


def _too_large_detail() -> str:
    limit_mb = settings.max_audio_upload_bytes // (1024 * 1024)
    return f"El audio supera el tamaño máximo permitido ({limit_mb} MB)."


def reject_oversized_body(request: Request) -> None:
    """Rechaza el request por `Content-Length` antes de leer el body.

    Ni FastAPI ni Starlette limitan el tamaño de un archivo subido: el
    `max_part_size` de Starlette solo aplica a los campos de texto del multipart
    (ver `starlette/formparsers.py`, `on_part_data`), así que el límite lo
    ponemos aquí.

    Al ser una dependency corre ANTES de que FastAPI parsee el multipart, de
    modo que un archivo enorme se corta sin volcarlo antes al disco temporal.
    `Content-Length` lo declara el cliente y puede mentir (o no venir, si sube
    en chunked): la comprobación definitiva es el conteo de bytes que hace
    `save_recording` al escribir.
    """
    header = request.headers.get("content-length")
    if header is None:
        return

    try:
        declared_bytes = int(header)
    except ValueError:
        # Header malformado: no es motivo para rechazar aquí. El conteo real
        # durante la escritura sigue protegiendo el límite.
        return

    if declared_bytes > settings.max_audio_upload_bytes + _MULTIPART_OVERHEAD_ALLOWANCE:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=_too_large_detail(),
        )


async def _analyze_best_effort(
    text: str,
    *,
    recording_id: str,
    uid: str,
    language: str | None,
    duration_seconds: float | None,
) -> AnalysisOutcome | None:
    """Calcula las métricas y pide el análisis. Devuelve `None` si no se pudo.

    Extraído a una función para que el endpoint no acabe siendo una escalera de
    `try` anidados: aquí dentro nada propaga, y quien llama solo tiene que
    saber si hay análisis o no.

    Devuelve el `AnalysisOutcome` completo, no solo el `RecordingAnalysis`,
    porque el `raw_response` que lleva dentro hace falta para guardar la fila.
    Mantener los dos juntos evita tener que pasárselos por separado, o peor,
    guardarlos en algún sitio compartido entre las dos funciones.
    """
    try:
        metrics = compute_metrics(
            text,
            duration_seconds=duration_seconds,
            language=language,
            # Solo para que el aviso de idioma desconocido se pueda rastrear
            # hasta la grabación concreta.
            recording_id=recording_id,
        )
        outcome = await analyze(text, metrics=metrics)
    except AnalysisError as exc:
        # Degradación PREVISTA: Groq no responde, se agotó la cuota (el nivel
        # on-demand se queda corto enseguida) o devolvió algo inservible.
        # WARNING y no ERROR porque el sistema se está comportando como se
        # diseñó.
        logger.warning(
            "Análisis no disponible: la grabación se devuelve sin análisis. "
            "recording_id=%s uid=%s motivo=%s",
            recording_id,
            uid,
            exc,
        )
        return None
    except Exception as exc:  # noqa: BLE001
        # Red de seguridad para un fallo NUESTRO: un bug en el cálculo de las
        # métricas o en el armado del análisis. Se registra como ERROR —esto sí
        # hay que arreglarlo, no es degradación esperada— pero tampoco rompe la
        # respuesta: la transcripción ya está hecha y pagada, y el usuario no
        # tiene por qué perderla por un error nuestro.
        logger.error(
            "Fallo inesperado al analizar la grabación (es un bug, revisar). "
            "recording_id=%s uid=%s tipo=%s motivo=%s",
            recording_id,
            uid,
            type(exc).__name__,
            exc,
        )
        return None

    logger.info(
        "Análisis generado: recording_id=%s uid=%s clarity=%s confidence=%s pace=%s",
        recording_id,
        uid,
        outcome.analysis.scores.clarity,
        outcome.analysis.scores.confidence,
        outcome.analysis.scores.pace,
    )
    return outcome


async def _persist_analysis_best_effort(
    outcome: AnalysisOutcome,
    *,
    recording_id: str,
    uid: str,
    recording_was_saved: bool,
) -> None:
    """Guarda el análisis. Ningún fallo aquí cambia la respuesta."""
    if not recording_was_saved:
        # `analyses.recording_id` es clave ajena: sin la fila de la grabación el
        # insert fallaría siempre. Se evita intentarlo para no dejar un WARNING
        # que culpe al análisis de un problema que era de la grabación.
        logger.warning(
            "El análisis no se guarda porque su grabación tampoco se guardó "
            "(ver el WARNING anterior). recording_id=%s uid=%s",
            recording_id,
            uid,
        )
        return

    try:
        await persist_analysis(
            recording_id=recording_id,
            user_uid=uid,
            analysis=outcome.analysis,
            raw_response=outcome.raw_response,
        )
    except AnalysisPersistenceError as exc:
        # Menos grave que perder la grabación: el texto sigue en la base y el
        # análisis se puede recalcular sobre él. De ahí que no se pida
        # intervención manual, solo quede constancia.
        #
        # Sin `exc_info`, por el mismo motivo que en el fallo de la grabación.
        logger.warning(
            "El análisis se devolvió al cliente pero no se guardó; se puede "
            "recalcular desde la transcripción. recording_id=%s uid=%s motivo=%s",
            recording_id,
            uid,
            exc,
        )


@router.post(
    "/recordings",
    response_model=RecordingUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Subir una grabación de audio",
    responses={
        401: {"description": "Falta el header Authorization, o el token es inválido/expirado."},
        413: {"description": "El audio supera el tamaño máximo permitido."},
        415: {"description": "El tipo de archivo no es un audio soportado."},
        422: {
            "description": (
                "Falta el campo `file` en el multipart/form-data, o Groq no pudo "
                "transcribir el audio (formato válido pero contenido ilegible)."
            )
        },
        503: {
            "description": (
                "Dependencia externa no disponible: no se pudieron obtener las "
                "claves públicas de Google, o Groq no está accesible o "
                "configurado (falta `GROQ_API_KEY`)."
            )
        },
    },
)
async def upload_recording(
    user: CurrentUserDep,
    # Declarado DESPUÉS de `user` a propósito: FastAPI resuelve las dependencies
    # en el orden de los parámetros, así que un request sin token se rechaza con
    # 401 sin llegar a mirar el tamaño ni a leer el body.
    _size_guard: Annotated[None, Depends(reject_oversized_body)],
    file: Annotated[
        UploadFile,
        File(description="Audio grabado en el móvil. expo-audio HIGH_QUALITY produce m4a/AAC."),
    ],
) -> RecordingUploadResponse:
    """Recibe el audio grabado en el móvil, lo transcribe, lo analiza y responde.

    El archivo se guarda primero en `settings.audio_storage_path` como
    `<uid>_<timestamp>_<aleatorio>.<ext>` —de forma que dos usuarios (o el mismo
    dos veces) nunca se pisan— y desde ahí se manda a Groq Whisper. La
    transcripción es síncrona: el móvil espera esta respuesta.

    De los cuatro pasos que vienen después de transcribir, NINGUNO puede
    cambiar el código de estado: guardar la grabación, calcular las métricas,
    pedir el análisis al modelo y guardarlo son todos best-effort. Lo único que
    se le puede negar al cliente con un error es la transcripción, porque es lo
    único que no se puede rehacer sin volver a pedirle que grabe.
    """
    try:
        saved = await save_recording(
            file,
            uid=user.uid,
            destination_dir=settings.audio_storage_path,
            max_bytes=settings.max_audio_upload_bytes,
        )
    except UnsupportedAudioTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Tipo de archivo no soportado: {exc.content_type!r}. "
                f"Tipos aceptados: {', '.join(sorted(ALLOWED_AUDIO_TYPES))}."
            ),
        ) from exc
    except AudioTooLargeError as exc:
        # Llega aquí cuando el `Content-Length` declaraba menos de lo que pesaba
        # el archivo de verdad.
        logger.warning("Audio rechazado por tamaño (uid=%s): %s", user.uid, exc)
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=_too_large_detail(),
        ) from exc

    logger.info(
        "Grabación guardada: %s (%d bytes, uid=%s)",
        saved.filename,
        saved.size_bytes,
        user.uid,
    )

    try:
        transcription = await transcribe(saved.path)
    except TranscriptionUnavailableError as exc:
        # Groq no está accesible o falta configurarlo: el audio del cliente
        # puede ser perfectamente válido, así que no se le culpa con un 4xx.
        logger.error("Transcripción no disponible (uid=%s): %s", user.uid, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo transcribir el audio en este momento.",
        ) from exc
    except TranscriptionFailedError as exc:
        # El formato era aceptable (ya pasó el filtro de `content_type`) pero el
        # contenido no se pudo transcribir: 422, no 415.
        logger.warning("Groq no pudo transcribir %s: %s", saved.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="El audio no se pudo transcribir. Comprueba la grabación.",
        ) from exc

    # Persistencia BEST-EFFORT: ningún fallo de aquí abajo cambia el 201.
    #
    # A estas alturas el audio ya se transcribió correctamente y esa llamada a
    # Groq ya se pagó. Devolverle un 5xx al móvil por un problema de nuestra
    # base de datos le haría perder una transcripción buena y probablemente
    # reintentar, pagándola otra vez. Así que el error se registra y se sigue.
    try:
        await persist_recording(
            recording_id=saved.recording_id,
            user_uid=user.uid,
            text=transcription.text,
            language=transcription.language,
            duration_seconds=transcription.duration_seconds,
            size_bytes=saved.size_bytes,
            content_type=saved.content_type,
            transcription_model=transcription.model,
        )
    except RecordingPersistenceError as exc:
        # El audio NO se borra: es lo único que queda de esta grabación, y con
        # él se puede rehacer la transcripción a mano. Mismo criterio que cuando
        # falla la transcripción.
        grabacion_guardada = False

        # El WARNING lleva a propósito solo identificadores y la ruta del
        # archivo, NUNCA el texto transcrito: son las palabras de una persona,
        # y los logs no son el sitio donde deben acabar.
        #
        # Y NO lleva `exc_info`. El mensaje de `RecordingPersistenceError` ya
        # viene saneado por `_motivo()` (ver `recording_repository.py`) con lo
        # accionable —clase del error y constraint que saltó— y sin datos del
        # usuario. El traceback, en cambio, arrastra la excepción original con
        # el `DETAIL:` de PostgreSQL, que en una violación de constraint
        # incluye la fila completa que falló. No compensa por lo poco que añade.
        logger.warning(
            "PERSISTENCIA FALLIDA (requiere intervención manual): la "
            "transcripción se devolvió al cliente pero no se guardó en la base "
            "de datos. recording_id=%s uid=%s audio_conservado=%s motivo=%s",
            saved.recording_id,
            user.uid,
            saved.path,
            exc,
        )
    else:
        grabacion_guardada = True

        # Solo se borra el audio cuando la grabación ya está a salvo en la DB.
        #
        # DEUDA TÉCNICA: no hay endpoint para reintentar ni limpieza periódica,
        # así que los audios de los intentos fallidos (transcripción o
        # guardado) se acumulan sin límite. Decisión consciente; revisar al
        # pasar el almacenamiento a un bucket.
        saved.path.unlink(missing_ok=True)

        logger.info(
            "Grabación transcrita, guardada y audio borrado: %s "
            "(%d caracteres, idioma=%s, uid=%s)",
            saved.filename,
            len(transcription.text),
            transcription.language,
            user.uid,
        )

    outcome = await _analyze_best_effort(
        transcription.text,
        recording_id=saved.recording_id,
        uid=user.uid,
        language=transcription.language,
        duration_seconds=transcription.duration_seconds,
    )

    if outcome is not None:
        await _persist_analysis_best_effort(
            outcome,
            recording_id=saved.recording_id,
            uid=user.uid,
            recording_was_saved=grabacion_guardada,
        )

    return RecordingUploadResponse(
        recording_id=saved.recording_id,
        filename=saved.filename,
        content_type=saved.content_type,
        size_bytes=saved.size_bytes,
        text=transcription.text,
        language=transcription.language,
        duration_seconds=transcription.duration_seconds,
        model=transcription.model,
        # Se devuelve lo calculado con independencia de si se guardó: el cliente
        # ya ha esperado por ello, y negárselo por un problema de nuestra base
        # de datos no arregla nada.
        analysis=outcome.analysis if outcome is not None else None,
    )
