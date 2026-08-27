"""Persistencia en disco de las grabaciones de audio recibidas.

Esta capa solo mueve bytes: valida el tipo declarado y el tamaño, y escribe el
archivo con un nombre único. No transcribe ni analiza nada (eso llegará con la
integración de Groq Whisper).

Lanza errores de dominio propios; traducirlos a respuestas HTTP es tarea del
router, igual que hace `app/core/dependencies.py` con los de Firebase.
"""

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fastapi import UploadFile
from fastapi.concurrency import run_in_threadpool

# Tipos MIME aceptados -> extensión con la que se guarda el archivo. La
# extensión sale de esta tabla y NO del `filename` que manda el cliente: así el
# nombre en disco nunca depende de una cadena que controla el móvil (que podría
# traer `../` o una extensión engañosa).
#
# expo-audio con `RecordingPresets.HIGH_QUALITY` graba m4a (AAC) tanto en iOS
# como en Android, pero el Content-Type que acaba enviando varía según la
# plataforma y la librería de subida; de ahí los cuatro alias que mapean a
# `.m4a`. mp3/wav están para poder probar el endpoint con curl, y webm cubre el
# fallback de Expo en web.
ALLOWED_AUDIO_TYPES: dict[str, str] = {
    "audio/m4a": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/mp4": ".m4a",
    "audio/aac": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/webm": ".webm",
}

# Tamaño de bloque al copiar a disco: ni se cargan los 25 MB enteros en RAM ni
# se hacen miles de escrituras diminutas.
CHUNK_SIZE = 1024 * 1024

# El uid de Firebase es alfanumérico, pero acaba formando parte de una ruta:
# se filtra igualmente todo lo que no sea seguro en un nombre de archivo.
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_-]")


class AudioStorageError(Exception):
    """Base de los errores de esta capa."""


class UnsupportedAudioTypeError(AudioStorageError):
    """El Content-Type declarado no está en la lista blanca de audio."""

    def __init__(self, content_type: str | None) -> None:
        self.content_type = content_type
        super().__init__(f"Tipo de audio no soportado: {content_type!r}")


class AudioTooLargeError(AudioStorageError):
    """El archivo recibido supera el límite de tamaño."""

    def __init__(self, limit_bytes: int) -> None:
        self.limit_bytes = limit_bytes
        super().__init__(f"El audio supera el límite de {limit_bytes} bytes")


@dataclass(frozen=True, slots=True)
class SavedRecording:
    """Resultado de guardar una grabación en disco."""

    recording_id: str
    filename: str
    path: Path
    size_bytes: int
    content_type: str


def resolve_audio_type(content_type: str | None) -> tuple[str, str]:
    """Valida el Content-Type declarado y devuelve `(tipo normalizado, extensión)`.

    Raises:
        UnsupportedAudioTypeError: el tipo falta o no está en `ALLOWED_AUDIO_TYPES`.
    """
    if not content_type:
        raise UnsupportedAudioTypeError(content_type)

    # El header puede traer parámetros: "audio/mp4; codecs=mp4a.40.2".
    normalized = content_type.split(";", 1)[0].strip().lower()
    extension = ALLOWED_AUDIO_TYPES.get(normalized)
    if extension is None:
        raise UnsupportedAudioTypeError(content_type)

    return normalized, extension


def build_recording_id(uid: str) -> str:
    """Genera un ID único de grabación: `<uid>_<timestamp>_<aleatorio>`.

    Cada parte cubre una colisión distinta: el uid separa usuarios, el timestamp
    UTC ordena las grabaciones de un mismo usuario, y el sufijo aleatorio
    resuelve dos subidas del mismo usuario en el mismo segundo.
    """
    safe_uid = _UNSAFE_FILENAME_CHARS.sub("", uid) or "anon"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{safe_uid}_{stamp}_{uuid.uuid4().hex[:8]}"


async def save_recording(
    upload: UploadFile,
    *,
    uid: str,
    destination_dir: Path,
    max_bytes: int,
) -> SavedRecording:
    """Guarda el audio subido y devuelve sus metadatos.

    El tamaño se cuenta mientras se escribe, no se confía en `Content-Length`:
    esta es la comprobación de tamaño definitiva. Si se supera el límite (o la
    escritura falla a medias) el archivo parcial se borra antes de propagar el
    error, para no dejar basura en disco.

    Raises:
        UnsupportedAudioTypeError: el Content-Type no es de audio permitido.
        AudioTooLargeError: el contenido real supera `max_bytes`.
    """
    content_type, extension = resolve_audio_type(upload.content_type)

    recording_id = build_recording_id(uid)
    filename = f"{recording_id}{extension}"
    destination_dir.mkdir(parents=True, exist_ok=True)
    path = destination_dir / filename

    size_bytes = 0
    completed = False
    try:
        # `open`/`write` son bloqueantes: van al threadpool para no congelar el
        # event loop mientras se escriben hasta 25 MB. (`upload.read` ya lo hace
        # por su cuenta cuando el archivo ha desbordado a disco.)
        handle = await run_in_threadpool(path.open, "wb")
        try:
            while chunk := await upload.read(CHUNK_SIZE):
                size_bytes += len(chunk)
                if size_bytes > max_bytes:
                    raise AudioTooLargeError(max_bytes)
                await run_in_threadpool(handle.write, chunk)
            completed = True
        finally:
            await run_in_threadpool(handle.close)
    finally:
        if not completed:
            path.unlink(missing_ok=True)

    return SavedRecording(
        recording_id=recording_id,
        filename=filename,
        path=path,
        size_bytes=size_bytes,
        content_type=content_type,
    )
