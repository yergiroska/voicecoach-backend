"""Endpoints de grabaciones de audio.

`POST /recordings` recibe el archivo del móvil, lo guarda en disco y lo
transcribe con Groq Whisper de forma síncrona: el cliente espera la
transcripción en la misma respuesta. Si la transcripción va bien el audio se
borra del disco; si falla se conserva (ver el comentario en `upload_recording`).

El análisis de la comunicación (muletillas, ritmo, claridad) se construirá sobre
la transcripción en una fase posterior.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.core.config import settings
from app.core.dependencies import CurrentUserDep
from app.models.recording import RecordingUploadResponse
from app.services.audio_storage import (
    ALLOWED_AUDIO_TYPES,
    AudioTooLargeError,
    UnsupportedAudioTypeError,
    save_recording,
)
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
    """Recibe el audio grabado en el móvil, lo transcribe y devuelve el texto.

    El archivo se guarda primero en `settings.audio_storage_path` como
    `<uid>_<timestamp>_<aleatorio>.<ext>` —de forma que dos usuarios (o el mismo
    dos veces) nunca se pisan— y desde ahí se manda a Groq Whisper. La
    transcripción es síncrona: el móvil espera esta respuesta.
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

    # Borrado solo tras éxito. Si la transcripción falla el archivo se queda en
    # disco a propósito, para poder depurar el fallo.
    #
    # DEUDA TÉCNICA: no hay endpoint para reintentar ni limpieza periódica, así
    # que los audios de intentos fallidos se acumulan sin límite. Decisión
    # consciente; revisar al añadir persistencia o almacenamiento en bucket.
    saved.path.unlink(missing_ok=True)

    logger.info(
        "Grabación transcrita y audio borrado: %s (%d caracteres, idioma=%s, uid=%s)",
        saved.filename,
        len(transcription.text),
        transcription.language,
        user.uid,
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
    )
