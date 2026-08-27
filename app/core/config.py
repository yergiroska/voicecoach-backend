"""Configuración de la aplicación.

Las variables se leen de un archivo `.env` (no versionado) o del entorno.
A medida que se añadan integraciones (Groq, PostgreSQL/pgvector, Firebase Auth),
sus credenciales se declaran aquí como campos de `Settings`.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Raíz del repo (app/core/config.py -> app/core -> app -> raíz). Se usa para
# resolver rutas relativas de settings sin depender del directorio desde el
# que se arranque uvicorn.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Settings globales cargados desde variables de entorno / .env."""

    app_name: str = "VoiceCoach AI"
    app_version: str = "0.1.0"

    # Firebase Auth. El Project ID es público (no es un secreto): identifica el
    # proyecto y se usa como `aud`/`iss` esperados al verificar los ID tokens.
    # No hace falta clave privada de servicio: la verificación se hace contra
    # las claves públicas de Google. Ver app/core/firebase_auth.py.
    firebase_project_id: str = "voicecoach-ai-9071f"

    # Tolerancia de desfase de reloj (segundos) al validar `iat`/`exp`. Evita
    # falsos "Token used too early" cuando el reloj local va ligeramente atrasado.
    firebase_clock_skew_seconds: int = 10

    # Cuánto cachear las claves públicas de Google si su respuesta no trae
    # `Cache-Control: max-age`. Con cabecera, manda la cabecera.
    firebase_certs_cache_ttl_seconds: int = 3600

    # ---- Almacenamiento local de audio ----
    # Carpeta donde se guardan las grabaciones subidas a POST /recordings. Es
    # almacenamiento temporal de desarrollo (gitignorado); en producción esto
    # pasará a un bucket. Si la ruta es relativa se resuelve desde la raíz del
    # repo, no desde el CWD.
    audio_storage_dir: Path = Path("storage/audio")

    # Tamaño máximo aceptado por grabación. 25 MB ≈ 25 min de m4a/AAC de
    # expo-audio (HIGH_QUALITY, ~1 MB/min) y coincide con el límite de archivo
    # de la API de Groq Whisper, el siguiente paso del pipeline: aceptar más de
    # lo que Groq admite no aportaría nada.
    max_audio_upload_bytes: int = 25 * 1024 * 1024

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def audio_storage_path(self) -> Path:
        """`audio_storage_dir` como ruta absoluta."""
        if self.audio_storage_dir.is_absolute():
            return self.audio_storage_dir
        return _PROJECT_ROOT / self.audio_storage_dir


settings = Settings()
