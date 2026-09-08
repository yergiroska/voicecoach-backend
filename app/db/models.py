"""Modelos declarativos de SQLAlchemy: el schema de la base de datos.

Ojo con la diferencia respecto a `app/models/`: allí viven los schemas de
Pydantic, que describen lo que entra y sale por HTTP. Aquí viven las tablas.
Son cosas distintas a propósito —la API no tiene por qué exponer todas las
columnas, ni con los mismos nombres— y mezclarlas acopla el contrato público
con el schema físico.

Este módulo define el schema; `app/core/database.py` gestiona la conexión y
`app/services/recording_repository.py` hace las consultas.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Double,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    SmallInteger,
    Text,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Plantillas de nombre para índices y constraints. Se fijan ANTES de crear la
# primera tabla porque, sin ellas, PostgreSQL bautiza los constraints por su
# cuenta y Alembic genera migraciones con nombres autogenerados: el día que
# haya que hacer un DROP CONSTRAINT en un downgrade, no se sabe cómo se llama.
# Con esto el nombre es derivable de la tabla y las columnas.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Base declarativa común a todos los modelos."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


# `TIMESTAMP WITH TIME ZONE`, nunca sin zona: el móvil puede estar en cualquier
# huso y una marca de tiempo ambigua es imposible de arreglar a posteriori.
_TimestampTz = DateTime(timezone=True)


class User(Base):
    """Usuario de la app, identificado por su uid de Firebase.

    Tabla mínima a propósito: existe para que `recordings.user_uid` tenga una
    clave ajena real (y para que la memoria vectorial por usuario de la Fase 7
    cuelgue de algo). La autenticación sigue siendo íntegramente de Firebase;
    esto no es un registro de credenciales.

    No se guarda el email aunque el ID token lo traiga: hoy nadie lo consume, es
    dato personal, y en Firebase puede ser nulo o cambiar. Se añadirá cuando
    haya preferencias de usuario que lo necesiten.
    """

    __tablename__ = "users"

    # El uid CRUDO del claim `sub`, sin sanear. Importante: el que va dentro de
    # `recordings.id` pasó por el filtro de caracteres de
    # `audio_storage.build_recording_id`, que es solo para nombres de archivo.
    # El uid de verdad es este.
    uid: Mapped[str] = mapped_column(Text, primary_key=True)

    created_at: Mapped[datetime] = mapped_column(
        _TimestampTz,
        nullable=False,
        # Default en el servidor, no en Python: así la hora la pone siempre el
        # mismo reloj (el de PostgreSQL), sin depender del huso del proceso.
        server_default=func.now(),
    )


class Recording(Base):
    """Una grabación ya transcrita.

    Es un HECHO consumado e inmutable: este audio se subió y produjo este texto.
    La interpretación de ese texto (muletillas, ritmo, claridad, confianza) vive
    en la tabla `analyses`, separada, para poder reanalizar sin tocar esto.

    La fila se escribe DESPUÉS de transcribir con éxito, así que no hay estados
    intermedios: si existe la fila, la transcripción salió bien.
    """

    __tablename__ = "recordings"

    # El `recording_id` que ya genera `audio_storage.build_recording_id()`:
    # `<uid_saneado>_<timestampUTC>_<8 hex>`. Se reutiliza como clave primaria
    # en lugar de inventar un UUID nuevo para tener UN SOLO identificador en
    # todo el sistema: el que el cliente recibe, el que sale en los logs y el
    # que está en la DB son el mismo. Además es ordenable y trazable de un
    # vistazo. El precio es una PK de texto (~150 bytes), irrelevante a esta
    # escala.
    id: Mapped[str] = mapped_column(Text, primary_key=True)

    user_uid: Mapped[str] = mapped_column(
        Text,
        # CASCADE: si algún día se borra un usuario (derecho al olvido), sus
        # grabaciones —y en cadena sus análisis— se van con él. Es voz de una
        # persona: no debe quedar huérfana en la tabla.
        ForeignKey("users.uid", ondelete="CASCADE"),
        nullable=False,
    )

    # Puede ser cadena vacía: un audio en silencio es un resultado válido y ya
    # está documentado así en `RecordingUploadResponse.text`. Vacío no es lo
    # mismo que NULL —hubo transcripción, y no encontró habla—, de ahí el
    # NOT NULL.
    text: Mapped[str] = mapped_column(Text, nullable=False)

    # `language` y `duration_seconds` son nullable porque Groq no los garantiza:
    # llegan como campos extra sin tipar en el SDK (ver `transcription_service`).
    language: Mapped[str | None] = mapped_column(Text, nullable=True)

    # `Double` -> DOUBLE PRECISION. La duración la usa el cálculo de palabras
    # por minuto de la Fase 6.2, que queda en `NULL` si esto viene vacío.
    duration_seconds: Mapped[float | None] = mapped_column(Double, nullable=True)

    # INTEGER basta: el límite de subida son 25 MB (`max_audio_upload_bytes`).
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)

    # Tipo MIME ya normalizado por `audio_storage.resolve_audio_type`.
    content_type: Mapped[str] = mapped_column(Text, nullable=False)

    # Qué modelo de Whisper produjo el texto. Se guarda porque cambiar de
    # `whisper-large-v3` a `-turbo` cambia la calidad de la transcripción y, con
    # ella, la de cualquier análisis construido encima.
    transcription_model: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        _TimestampTz,
        nullable=False,
        server_default=func.now(),
    )


# Índice compuesto para "las grabaciones de este usuario, de la más reciente a
# la más antigua": es la consulta del histórico y también la que necesitará la
# línea base personal de la Fase 7. Al empezar por `user_uid` sirve además como
# índice de la clave ajena (PostgreSQL no las indexa solo).
Index(
    "ix_recordings_user_uid_created_at",
    Recording.user_uid,
    Recording.created_at.desc(),
)


class Analysis(Base):
    """El análisis de comunicación de una grabación.

    Tabla separada de `recordings` por una razón concreta: una grabación es un
    HECHO (este audio produjo este texto) y un análisis es una INTERPRETACIÓN
    de ese hecho, que depende del prompt y del modelo usados. Separarlos permite
    reanalizar un audio sin tocar el registro original, y guardar varios
    análisis del mismo audio a la vez.

    Eso solo funciona si cada fila dice CÓMO se produjo, de ahí
    `analysis_model` y `prompt_version`: sin ellos dos análisis del mismo audio
    son incomparables y no se puede distinguir "el usuario mejoró" de "cambié
    el prompt". Son la razón de ser de esta tabla, no metadatos decorativos.

    Ojo al leer los scores: son juicios de un LLM con un ruido medido de unos
    ±5 puntos incluso con `temperature=0` (ver el docstring de
    `app/services/analysis_service.py`). Cualquier consulta que compare
    progreso debe mirar tendencias sobre varias grabaciones, nunca dos filas.
    """

    __tablename__ = "analyses"

    # UUID generado por PostgreSQL. `gen_random_uuid()` es nativa desde
    # PostgreSQL 13, así que no hace falta la extensión pgcrypto.
    #
    # Aquí sí un id sintético, al contrario que en `recordings` (que reutiliza
    # el `recording_id` de texto): un análisis no tiene ningún identificador
    # natural —el mismo audio puede tener varios— y no viaja al cliente.
    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )

    recording_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("recordings.id", ondelete="CASCADE"),
        nullable=False,
    )

    # DENORMALIZADO a propósito: se puede llegar al uid por `recordings`, pero
    # la línea base de la Fase 7 va a consultar constantemente "todos los
    # análisis de este usuario, por fecha", y con esta columna eso es un
    # recorrido de índice en vez de un JOIN en cada consulta.
    #
    # El riesgo de dos fuentes para el mismo dato es bajo porque
    # `recordings.user_uid` es inmutable: una grabación no cambia de dueño.
    user_uid: Mapped[str] = mapped_column(
        Text,
        ForeignKey("users.uid", ondelete="CASCADE"),
        nullable=False,
    )

    # ---- Métricas deterministas (calculadas en Python) ----
    # NOT NULL porque siempre se pueden calcular: son las que hacen que un
    # análisis degradado (LLM caído) siga teniendo valor.
    word_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # Nullable: necesita la duración del audio, que Groq no garantiza.
    words_per_minute: Mapped[float | None] = mapped_column(Double, nullable=True)

    filler_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # Lista de `{"word": ..., "count": ...}`, ya ordenada de más frecuente a
    # menos. JSONB y no una tabla aparte: es un desglose que solo se lee junto
    # a su análisis, nunca se consulta por sí mismo.
    filler_words: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
    )

    # Si la detección de muletillas se ejecutó. Es `false` cuando el idioma no
    # era español (la lista de muletillas es es-ES).
    #
    # ESTA COLUMNA NO ESTABA EN EL DISEÑO INICIAL y se añade por necesidad: sin
    # ella, al leer una fila con `filler_count = 0` sería imposible saber si no
    # había muletillas o si no se buscaron. Es exactamente la ambigüedad que el
    # campo `SpeechMetrics.fillers_analyzed` existe para evitar, y no
    # persistirla la reintroduciría en cuanto alguien consultase el histórico.
    fillers_analyzed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
    )

    # ---- Juicios del modelo de lenguaje ----
    # Los tres son nullable, y `NULL` NO es cero: significa "el modelo no dio
    # un valor usable". Un score fuera de rango se descarta antes de llegar
    # aquí (ver `_score_valido` en app/models/analysis.py), así que se prefiere
    # guardar la fila con las métricas buenas y este campo vacío antes que
    # perder el análisis entero.
    #
    # SMALLINT sobra para 0-100 y ocupa la mitad que un INTEGER.
    clarity_score: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    confidence_score: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    pace_score: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)

    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Lista de strings. Vacía si el modelo no propuso ninguna.
    suggestions: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
    )

    # ---- Trazabilidad ----
    # La respuesta del modelo tal cual, sin filtrar. Es el seguro de vida de
    # esta tabla: si mañana se quiere exponer una métrica que el modelo ya
    # devolvía pero no se extraía, está aquí y no hay que reanalizar (ni volver
    # a pagar) los audios antiguos.
    #
    # Cuando no se llama al modelo (audio en silencio) guarda el motivo, p. ej.
    # `{"skipped": "empty_transcription"}`: así una consulta puede separar "el
    # modelo no valoró" de "no se le preguntó, y por qué".
    raw_response: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    analysis_model: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_version: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        _TimestampTz,
        nullable=False,
        server_default=func.now(),
    )

    # Los rangos se comprueban también en la base de datos, no solo en Pydantic.
    # Pydantic protege la API; esto protege la TABLA de cualquier otra vía de
    # escritura (un script de migración de datos, una corrección a mano en
    # psql). Un score fuera de 0-100 corrompería la línea base de la Fase 7 sin
    # que nada fallara, así que conviene que la propia columna lo rechace.
    __table_args__ = (
        CheckConstraint(
            "clarity_score BETWEEN 0 AND 100", name="clarity_score_range"
        ),
        CheckConstraint(
            "confidence_score BETWEEN 0 AND 100", name="confidence_score_range"
        ),
        CheckConstraint("pace_score BETWEEN 0 AND 100", name="pace_score_range"),
        # `filler_count` es una suma de conteos y `word_count` un total: ninguno
        # puede ser negativo.
        CheckConstraint("word_count >= 0", name="word_count_non_negative"),
        CheckConstraint("filler_count >= 0", name="filler_count_non_negative"),
    )


# "El último análisis de esta grabación" y el histórico de reanálisis.
Index(
    "ix_analyses_recording_id_created_at",
    Analysis.recording_id,
    Analysis.created_at.desc(),
)

# "Todos los análisis de este usuario, del más reciente al más antiguo": es la
# consulta que alimentará la línea base personal de la Fase 7, y la razón de que
# `user_uid` esté denormalizado en esta tabla.
Index(
    "ix_analyses_user_uid_created_at",
    Analysis.user_uid,
    Analysis.created_at.desc(),
)

# Sin `relationship()` a propósito. Con SQLAlchemy async, acceder a una relación
# no cargada lanza MissingGreenlet en vez de emitir la consulta, así que las
# relaciones hay que declararlas junto al `selectinload` que las llene. Se
# añadirán cuando un endpoint de lectura lo necesite, no antes.
