"""Schemas Pydantic de las grabaciones de audio."""

from pydantic import BaseModel, Field


class RecordingUploadResponse(BaseModel):
    """Respuesta de `POST /recordings`: confirma que el audio quedó guardado.

    Todavía no incluye transcripción ni análisis: en esta fase el endpoint solo
    recibe y persiste el archivo.
    """

    recording_id: str = Field(
        description=(
            "Identificador único de la grabación. El cliente lo guarda para "
            "consultar la transcripción y el análisis cuando existan."
        ),
        examples=["Ab3xY9kLmN0pQrS1tUvW2xYz_20260827T105412Z_9f2c1d7e"],
    )
    filename: str = Field(
        description="Nombre con el que el backend guardó el archivo.",
        examples=["Ab3xY9kLmN0pQrS1tUvW2xYz_20260827T105412Z_9f2c1d7e.m4a"],
    )
    content_type: str = Field(
        description="Tipo MIME del audio, normalizado (sin parámetros).",
        examples=["audio/m4a"],
    )
    size_bytes: int = Field(
        description="Tamaño real del archivo recibido, en bytes.",
        examples=[184320],
    )
