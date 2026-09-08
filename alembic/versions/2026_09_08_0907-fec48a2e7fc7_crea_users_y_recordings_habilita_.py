"""crea users y recordings, habilita pgvector

Primera migración del proyecto: hasta ahora VoiceCoach AI era stateless.

Crea las dos tablas de la Fase 6.1 (`users` y `recordings`) y habilita la
extensión `vector`. La tabla `analyses` llega en la migración siguiente, junto
con el análisis de IA.

Revision ID: fec48a2e7fc7
Revises:
Create Date: 2026-09-08

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "fec48a2e7fc7"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # pgvector se habilita YA, aunque ninguna columna lo use todavía. Es
    # idempotente y gratis, y así la memoria vectorial por usuario de la Fase 7
    # es "añadir una tabla" en vez de "descubrir un problema de permisos en
    # producción": habilitar una extensión requiere privilegios que un usuario
    # de aplicación puede no tener, y es mejor enterarse ahora.
    #
    # En local (PostgreSQL 18 con pgvector 0.8.2 ya instalado) esto la crea; en
    # Supabase la extensión suele venir habilitada y el IF NOT EXISTS lo
    # convierte en un no-op.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "users",
        # uid CRUDO del claim `sub` de Firebase, sin sanear.
        sa.Column("uid", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("uid", name="pk_users"),
    )

    op.create_table(
        "recordings",
        # El `recording_id` que genera audio_storage.build_recording_id().
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("user_uid", sa.Text(), nullable=False),
        # Cadena vacía permitida (audio en silencio); NULL no.
        sa.Column("text", sa.Text(), nullable=False),
        # Groq no garantiza `language` ni `duration`: nullable.
        sa.Column("language", sa.Text(), nullable=True),
        sa.Column("duration_seconds", sa.Double(), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("content_type", sa.Text(), nullable=False),
        sa.Column("transcription_model", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_recordings"),
        sa.ForeignKeyConstraint(
            ["user_uid"],
            ["users.uid"],
            name="fk_recordings_user_uid_users",
            # Borrar un usuario se lleva sus grabaciones (y, en cadena, sus
            # análisis): es voz de una persona, no debe quedar huérfana.
            ondelete="CASCADE",
        ),
    )

    # "Las grabaciones de este usuario, de la más reciente a la más antigua":
    # el histórico y, sobre todo, la línea base personal de la Fase 7. Al
    # empezar por `user_uid` sirve además como índice de la clave ajena, que
    # PostgreSQL no crea por su cuenta.
    op.create_index(
        "ix_recordings_user_uid_created_at",
        "recordings",
        ["user_uid", sa.text("created_at DESC")],
    )


def downgrade() -> None:
    op.drop_index("ix_recordings_user_uid_created_at", table_name="recordings")
    op.drop_table("recordings")
    op.drop_table("users")

    # La extensión `vector` NO se elimina a propósito. `DROP EXTENSION` es
    # destructivo más allá del alcance de esta migración (se llevaría cualquier
    # columna `vector` de otro sitio) y en una DB gestionada como Supabase puede
    # estar habilitada de fábrica: revertir esta migración no debería
    # deshabilitarla.
