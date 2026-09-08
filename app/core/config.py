"""Configuración de la aplicación.

Las variables se leen de un archivo `.env` (no versionado) o del entorno.
A medida que se añadan integraciones (PostgreSQL/pgvector), sus credenciales se
declaran aquí como campos de `Settings`.
"""

from pathlib import Path

from pydantic import SecretStr
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

    # ---- Groq (transcripción con Whisper) ----
    # PRIMER secreto real del proyecto: a diferencia del Project ID de Firebase,
    # esta clave no es pública. Va en `.env` (gitignorado), nunca en el repo.
    #
    # Es opcional a propósito (`| None`, sin default útil): si fuese obligatoria,
    # pydantic lanzaría ValidationError al importar este módulo y el proceso no
    # arrancaría, tumbando también /health y /me. En su lugar la app arranca con
    # un WARNING (ver el `lifespan` de app/main.py) y solo POST /recordings
    # responde 503. `SecretStr` evita que la clave aparezca en logs o en un
    # `repr(settings)`.
    groq_api_key: SecretStr | None = None

    # Modelo de transcripción. `whisper-large-v3` prioriza precisión (10,3% WER)
    # frente a `whisper-large-v3-turbo` (12% WER, más rápido y barato), porque el
    # análisis de muletillas y ritmo depende de la calidad de la transcripción.
    # Configurable para poder comparar ambos sin tocar código.
    groq_whisper_model: str = "whisper-large-v3"

    # Corte de la llamada a Groq. Con transcripción síncrona el móvil espera esta
    # respuesta, así que sin timeout explícito una llamada colgada bloquearía el
    # request indefinidamente. 120 s cubre subir a Groq y transcribir un audio
    # cercano al límite de 25 MB.
    groq_timeout_seconds: float = 120.0

    # ---- Groq (análisis de comunicación con un LLM) ----
    # Modelo que valora claridad, confianza y ritmo. Usa la misma GROQ_API_KEY.
    #
    # `openai/gpt-oss-120b` elegido con datos, no por defecto (ver la
    # comparativa en el docstring de app/services/analysis_service.py):
    #   - es uno de los tres modelos de la cuenta que admiten salida
    #     estructurada ESTRICTA, que es lo que garantiza un JSON con la forma
    #     de `AnalysisLlmOutput`;
    #   - `qwen/qwen3.8-27b` quedó descartado porque puntuaba sobre 10 en vez de
    #     sobre 100 pese al esquema, y cuesta 5 veces más;
    #   - `openai/gpt-oss-20b` es la mitad de precio pero repite una sugerencia
    #     incoherente ("revisa la transcripción antes de grabar": no entiende
    #     que la transcripción es POSTERIOR a la grabación), y las sugerencias
    #     son justo el valor que ve el usuario.
    #
    # Configurable para poder comparar modelos sin tocar código, igual que
    # `groq_whisper_model`.
    groq_analysis_model: str = "openai/gpt-oss-120b"

    # Corte de la llamada de análisis, separado del de transcripción y mucho más
    # corto a propósito: medido, el análisis tarda 1-2 s. Cuando se ejecuta, el
    # audio YA está transcrito, así que heredar los 120 s de Whisper significaría
    # tener al móvil esperando dos minutos por un extra que además es opcional
    # (si falla, la respuesta sale con `analysis: null`).
    groq_analysis_timeout_seconds: float = 30.0

    # ---- Persistencia: PostgreSQL ----
    # SEGUNDO secreto del proyecto (lleva la contraseña de la DB dentro de la
    # URL), de ahí `SecretStr`: sin él la cadena entera —contraseña incluida—
    # aparecería en cualquier log que volcase la config o en un `repr(settings)`.
    #
    # Opcional por el mismo motivo que `groq_api_key`: hacerla obligatoria
    # lanzaría ValidationError al importar este módulo y tumbaría /health y /me,
    # que no necesitan base de datos. Sin ella la app arranca con un WARNING y
    # POST /recordings sigue transcribiendo: simplemente no persiste la fila
    # (ver la decisión de "insert best-effort" en app/routers/recordings.py).
    #
    # No se le pone un default con credenciales locales a propósito: un
    # usuario/contraseña inventados en el código son una trampa (parece que
    # funciona hasta que no funciona). Se declara en `.env` (ver .env.example).
    database_url: SecretStr | None = None

    # Volcar al log cada sentencia SQL que ejecuta SQLAlchemy. Útil al depurar
    # migraciones o consultas; muy ruidoso, así que por defecto apagado.
    database_echo: bool = False

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

    @property
    def groq_configured(self) -> bool:
        """Si hay una clave de Groq usable.

        No basta con `groq_api_key is not None`: `.env.example` trae la línea
        `GROQ_API_KEY=` vacía, así que copiarlo a `.env` sin rellenarla da un
        `SecretStr('')` —presente pero inservible—. Se trata igual que ausente.
        """
        if self.groq_api_key is None:
            return False
        return bool(self.groq_api_key.get_secret_value().strip())

    @property
    def database_configured(self) -> bool:
        """Si hay una URL de base de datos usable.

        Mismo criterio que `groq_configured`: una variable presente pero vacía
        (`DATABASE_URL=` copiado de `.env.example` sin rellenar) cuenta como no
        configurada, no como URL inválida.
        """
        if self.database_url is None:
            return False
        return bool(self.database_url.get_secret_value().strip())

    @property
    def database_dsn(self) -> str:
        """La URL de la DB como string plano, con el driver async garantizado.

        SQLAlchemy necesita el string sin envolver, así que aquí se saca del
        `SecretStr`. Este es el único sitio que lo desenvuelve: el resto del
        código pide `settings.database_dsn`, nunca `settings.database_url`.

        Además normaliza el esquema. `create_async_engine` exige un driver
        asíncrono explícito (`postgresql+asyncpg://`), pero las URLs que dan los
        proveedores vienen como `postgresql://` (Supabase) o `postgres://`
        (estilo Heroku, que SQLAlchemy directamente no acepta). Se reescriben
        aquí para poder pegar la cadena del proveedor en `.env` tal cual, sin
        acordarse de este detalle. Una URL que ya trae driver (`+asyncpg`, o
        `+psycopg` para un script puntual) se respeta sin tocar.

        Raises:
            ValueError: no hay URL configurada, o no tiene forma de URL.
        """
        if not self.database_configured:
            raise ValueError(
                "DATABASE_URL no configurada: no hay base de datos a la que conectarse."
            )

        # `database_configured` ya garantiza que no es None ni está vacía.
        raw = self.database_url.get_secret_value().strip()  # type: ignore[union-attr]

        scheme, separator, rest = raw.partition("://")
        if not separator:
            raise ValueError(
                "DATABASE_URL no parece una URL: falta el '://' tras el esquema."
            )

        if scheme in ("postgres", "postgresql"):
            return f"postgresql+asyncpg://{rest}"

        return raw


settings = Settings()
