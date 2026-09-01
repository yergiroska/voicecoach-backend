"""Transcripción de audio con Groq Whisper.

Recibe la ruta de un audio ya guardado en disco y devuelve su transcripción.
No decide nada sobre el ciclo de vida del archivo (no lo borra) ni sabe de HTTP:
lanza errores de dominio y el router los traduce, igual que hacen
`app/services/audio_storage.py` y `app/core/firebase_auth.py`.

La distinción entre los dos errores de esta capa es la parte importante y viene
copiada de `firebase_auth.py`: separar "el cliente mandó algo que no vale" de
"el servicio del que dependemos no está disponible". Confundirlos hace que un
fallo de configuración nuestro parezca culpa del móvil.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from fastapi.concurrency import run_in_threadpool
from groq import (
    APIConnectionError,
    APIStatusError,
    AsyncGroq,
    AuthenticationError,
    RateLimitError,
)

from app.core.config import settings

logger = logging.getLogger(__name__)

# `verbose_json` es obligatorio para obtener `language` y `duration`: con `json`
# a secas la respuesta solo trae `text`.
_RESPONSE_FORMAT = "verbose_json"

# Cliente cacheado a nivel de módulo. Deliberadamente NO se crea en el import:
# así la app arranca sin GROQ_API_KEY y solo falla este endpoint (ver el
# `lifespan` de app/main.py).
_client: AsyncGroq | None = None


class TranscriptionError(Exception):
    """Base de los errores de esta capa."""


class TranscriptionFailedError(TranscriptionError):
    """Groq rechazó este audio concreto: su contenido no se pudo transcribir.

    Esto sí es sobre lo que envió el cliente (audio corrupto, ilegible...), así
    que el router lo traduce a 4xx.
    """


class TranscriptionUnavailableError(TranscriptionError):
    """No se pudo hablar con Groq, o falta configuración del servidor.

    Nunca es culpa del cliente: su petición puede ser perfectamente válida. El
    router lo traduce a 503, no a 401 ni a 500.
    """


@dataclass(frozen=True, slots=True)
class Transcription:
    """Resultado de transcribir un audio."""

    text: str
    language: str | None
    duration_seconds: float | None
    model: str


def _as_float(value: object) -> float | None:
    """Convierte a float lo que venga, o `None` si no se puede.

    Blinda contra cambios en el payload de Groq: `duration` no está tipado en el
    SDK (ver `transcribe`), así que podría llegar en un formato inesperado y no
    merece tumbar la petición con un 500.
    """
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        logger.warning("Groq devolvió una `duration` no numérica: %r", value)
        return None


def _get_client() -> AsyncGroq:
    """Devuelve el cliente de Groq, creándolo en la primera llamada.

    Raises:
        TranscriptionUnavailableError: no hay `GROQ_API_KEY` usable configurada.
    """
    global _client

    if _client is not None:
        return _client

    api_key = settings.groq_api_key
    if api_key is None or not settings.groq_configured:
        raise TranscriptionUnavailableError(
            "GROQ_API_KEY no configurada en el servidor."
        )

    _client = AsyncGroq(api_key=api_key.get_secret_value())
    return _client


async def transcribe(
    path: Path,
    *,
    model: str | None = None,
    timeout: float | None = None,
) -> Transcription:
    """Transcribe el audio de `path` con Groq Whisper.

    Args:
        path: audio ya guardado en disco. Su extensión debe estar entre los
            formatos que acepta Groq (flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav,
            webm); `ALLOWED_AUDIO_TYPES` de `audio_storage` ya es un subconjunto.
        model: modelo a usar. Por defecto `settings.groq_whisper_model`.
        timeout: segundos antes de cortar. Por defecto
            `settings.groq_timeout_seconds`.

    Raises:
        TranscriptionFailedError: Groq rechazó el audio (4xx que no sea de auth
            ni de rate limit).
        TranscriptionUnavailableError: falta la clave, Groq no está accesible,
            devolvió 5xx, limitó la petición o la clave no es válida.
    """
    client = _get_client()
    model = model or settings.groq_whisper_model
    timeout = timeout if timeout is not None else settings.groq_timeout_seconds

    # `read_bytes` bloquea y esto corre en el event loop. Se manda el contenido
    # en memoria (como mucho `max_audio_upload_bytes`, 25 MB) en lugar de un
    # handle abierto, para no meter E/S bloqueante dentro de httpx.
    try:
        content = await run_in_threadpool(path.read_bytes)
    except OSError as exc:
        # El archivo lo acabamos de escribir nosotros: si no se puede leer, el
        # problema es del servidor, no del audio que mandó el cliente.
        logger.error("No se pudo leer el audio para transcribir (%s): %s", path, exc)
        raise TranscriptionUnavailableError(
            "No se pudo leer el audio guardado."
        ) from exc

    try:
        response = await client.audio.transcriptions.create(
            file=(path.name, content),
            model=model,
            response_format=_RESPONSE_FORMAT,
            timeout=timeout,
        )
    except AuthenticationError as exc:
        # Un 401 de Groq significa que NUESTRA clave es mala. Propagarlo como
        # 401 al móvil provocaría logouts en cascada por un fallo de
        # configuración nuestro; mismo razonamiento que en firebase_auth.py.
        logger.error("Groq rechazó la API key: %s", exc)
        raise TranscriptionUnavailableError(
            "La API key de Groq no es válida."
        ) from exc
    except RateLimitError as exc:
        # Transitorio y ajeno al usuario: no ha hecho nada mal.
        logger.warning("Groq está limitando las peticiones: %s", exc)
        raise TranscriptionUnavailableError(
            "Groq está limitando las peticiones."
        ) from exc
    except APIConnectionError as exc:
        # Cubre también APITimeoutError, que hereda de esta.
        logger.error("No se pudo conectar con Groq: %s", exc)
        raise TranscriptionUnavailableError("No se pudo contactar con Groq.") from exc
    except APIStatusError as exc:
        # AuthenticationError y RateLimitError también heredan de APIStatusError,
        # pero ya se capturaron arriba: aquí quedan el resto de 4xx y los 5xx.
        if exc.status_code >= 500:
            logger.error("Groq devolvió un %s: %s", exc.status_code, exc)
            raise TranscriptionUnavailableError(
                "Groq devolvió un error de servidor."
            ) from exc

        logger.warning("Groq rechazó el audio (HTTP %s): %s", exc.status_code, exc)
        raise TranscriptionFailedError(
            f"Groq no pudo procesar el audio (HTTP {exc.status_code})."
        ) from exc

    # `language` y `duration` llegan como campos EXTRA sin declarar: el modelo de
    # respuesta del SDK 1.7.0 solo tipa `text`, y su BaseModel usa
    # `extra="allow"`. Por eso se leen con getattr y el dataclass los admite
    # `None`: si Groq deja de mandarlos, esto sigue funcionando.
    return Transcription(
        # Whisper devuelve el texto con un espacio inicial (" Bueno, eh...").
        # Se limpia aquí, en la capa que habla con Groq, para que ni el router ni
        # el móvil tengan que conocer ese detalle del proveedor. Un audio en
        # silencio pasa así de " " a "": cadena vacía, que ya es un resultado
        # válido y documentado en `RecordingUploadResponse.text`.
        text=response.text.strip(),
        language=getattr(response, "language", None),
        duration_seconds=_as_float(getattr(response, "duration", None)),
        model=model,
    )
