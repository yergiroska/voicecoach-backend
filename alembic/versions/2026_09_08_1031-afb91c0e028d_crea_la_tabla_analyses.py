"""crea la tabla analyses

Segunda migración: la interpretación de las transcripciones.

`analyses` va aparte de `recordings` porque son dos cosas distintas: una
grabación es un hecho inmutable (este audio produjo este texto) y un análisis es
una lectura de ese hecho, que depende del prompt y del modelo. Separarlas
permite reanalizar sin tocar el registro original y guardar varios análisis del
mismo audio.

Generada con `alembic revision --autogenerate` contra la base de desarrollo. El
diff detectó únicamente esta tabla y sus dos índices, sin ninguna diferencia en
`users` ni `recordings`: prueba de que la migración 0001, escrita a mano,
coincide con los modelos.

Revision ID: afb91c0e028d
Revises: fec48a2e7fc7
Create Date: 2026-09-08

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "afb91c0e028d"
down_revision: str | None = "fec48a2e7fc7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "analyses",
        # UUID sintético: un análisis no tiene identificador natural (el mismo
        # audio puede tener varios) y no viaja al cliente. `gen_random_uuid()`
        # es nativa desde PostgreSQL 13, sin necesidad de pgcrypto.
        sa.Column(
            "id",
            sa.Uuid(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("recording_id", sa.Text(), nullable=False),
        # Denormalizado: se podría llegar por `recordings`, pero la línea base
        # de la Fase 7 consultará "todos los análisis de este usuario" sin
        # parar, y así es un recorrido de índice en vez de un JOIN.
        sa.Column("user_uid", sa.Text(), nullable=False),
        # ---- Métricas deterministas (Python). NOT NULL: siempre calculables.
        sa.Column("word_count", sa.Integer(), nullable=False),
        # Nullable: necesita la duración del audio, que Groq no garantiza.
        sa.Column("words_per_minute", sa.Double(), nullable=True),
        sa.Column("filler_count", sa.Integer(), nullable=False),
        # Lista de {"word": ..., "count": ...}, ya ordenada.
        sa.Column(
            "filler_words",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        # Sin esta columna, un `filler_count = 0` en el histórico sería
        # ambiguo: ¿no había muletillas, o no se buscaron porque el audio no
        # era en español? El default `true` es correcto para las filas nuevas
        # de un producto que hoy solo analiza español.
        sa.Column(
            "fillers_analyzed",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        # ---- Juicios del LLM. Nullable, y NULL no es cero: es "sin valorar".
        # SMALLINT sobra para 0-100 y ocupa la mitad que un INTEGER.
        sa.Column("clarity_score", sa.SmallInteger(), nullable=True),
        sa.Column("confidence_score", sa.SmallInteger(), nullable=True),
        sa.Column("pace_score", sa.SmallInteger(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column(
            "suggestions",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        # ---- Trazabilidad.
        # La respuesta del modelo sin filtrar: si mañana se quiere exponer un
        # dato que ya venía pero no se extraía, está aquí y no hay que
        # reanalizar (ni volver a pagar) los audios antiguos. Cuando no se
        # llama al modelo guarda el motivo, p. ej. {"skipped": "..."}.
        sa.Column(
            "raw_response", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        # Sin estas dos, dos análisis del mismo audio son incomparables y no se
        # puede distinguir "el usuario mejoró" de "cambié el prompt".
        sa.Column("analysis_model", sa.Text(), nullable=False),
        sa.Column("prompt_version", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        # Los rangos se validan también en la base de datos, no solo en
        # Pydantic: Pydantic protege la API, esto protege la TABLA de cualquier
        # otra vía de escritura (un script, una corrección a mano en psql). Un
        # score fuera de 0-100 corrompería la línea base sin que nada fallara.
        sa.CheckConstraint(
            "clarity_score BETWEEN 0 AND 100",
            name=op.f("ck_analyses_clarity_score_range"),
        ),
        sa.CheckConstraint(
            "confidence_score BETWEEN 0 AND 100",
            name=op.f("ck_analyses_confidence_score_range"),
        ),
        sa.CheckConstraint(
            "pace_score BETWEEN 0 AND 100",
            name=op.f("ck_analyses_pace_score_range"),
        ),
        sa.CheckConstraint(
            "filler_count >= 0", name=op.f("ck_analyses_filler_count_non_negative")
        ),
        sa.CheckConstraint(
            "word_count >= 0", name=op.f("ck_analyses_word_count_non_negative")
        ),
        # CASCADE en las dos: borrar un usuario se lleva sus grabaciones y, en
        # cadena, sus análisis. Es voz de una persona, no debe quedar huérfana.
        sa.ForeignKeyConstraint(
            ["recording_id"],
            ["recordings.id"],
            name=op.f("fk_analyses_recording_id_recordings"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_uid"],
            ["users.uid"],
            name=op.f("fk_analyses_user_uid_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_analyses")),
    )

    # "El último análisis de esta grabación" y el histórico de reanálisis.
    #
    # `literal_column("created_at DESC")` en vez del nombre a secas: es lo que
    # conserva el orden descendente del índice. Sin él, PostgreSQL crearía un
    # índice ascendente que no sirve igual para "los más recientes primero".
    op.create_index(
        "ix_analyses_recording_id_created_at",
        "analyses",
        ["recording_id", sa.literal_column("created_at DESC")],
        unique=False,
    )

    # "Todos los análisis de este usuario, del más reciente al más antiguo":
    # la consulta de la línea base de la Fase 7, y la razón de que `user_uid`
    # esté denormalizado en esta tabla.
    op.create_index(
        "ix_analyses_user_uid_created_at",
        "analyses",
        ["user_uid", sa.literal_column("created_at DESC")],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_analyses_user_uid_created_at", table_name="analyses")
    op.drop_index("ix_analyses_recording_id_created_at", table_name="analyses")
    op.drop_table("analyses")
