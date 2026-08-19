"""Configuración de la aplicación.

Las variables se leen de un archivo `.env` (no versionado) o del entorno.
A medida que se añadan integraciones (Groq, PostgreSQL/pgvector, Firebase Auth),
sus credenciales se declaran aquí como campos de `Settings`.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
