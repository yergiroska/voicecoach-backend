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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
