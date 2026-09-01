"""Schemas Pydantic de las grabaciones de audio."""

from pydantic import BaseModel, Field


class RecordingUploadResponse(BaseModel):
    """Respuesta de `POST /recordings`: el audio recibido y ya transcrito.

    La transcripción es síncrona, así que llega en esta misma respuesta. El
    archivo de audio se borra del disco en cuanto se transcribe con éxito, de
    modo que los campos `filename`/`size_bytes` describen algo que ya no existe
    cuando el cliente lee esto: se mantienen porque le sirven para verificar
    que llegó lo que envió.

    Todavía no incluye el análisis de la comunicación (muletillas, ritmo,
    claridad): eso se construirá sobre `text` en una fase posterior.
    """

    recording_id: str = Field(
        description=(
            "Identificador único de la grabación. Por ahora no hay persistencia "
            "detrás (ni fila en base de datos ni audio guardado): sirve para "
            "correlacionar esta respuesta con los logs del servidor."
        ),
        examples=["Ab3xY9kLmN0pQrS1tUvW2xYz_20260827T105412Z_9f2c1d7e"],
    )
    filename: str = Field(
        description=(
            "Nombre con el que el backend guardó el archivo mientras lo "
            "procesaba. Tras una transcripción correcta el archivo ya no existe."
        ),
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
    text: str = Field(
        description=(
            "Transcripción del audio. Puede venir vacía si la grabación es "
            "silencio o no contiene habla reconocible."
        ),
        examples=["Bueno, eh, hoy quiero hablaros de, o sea, cómo comunicamos."],
    )
    language: str | None = Field(
        default=None,
        description=(
            "Idioma detectado por Whisper. Es `null` si Groq no lo informa: no "
            "viene tipado en el SDK, así que se trata como opcional."
        ),
        examples=["Spanish"],
    )
    duration_seconds: float | None = Field(
        default=None,
        description=(
            "Duración del audio según Whisper, en segundos. `null` si Groq no "
            "lo informa. Útil para el análisis de ritmo (palabras por minuto)."
        ),
        examples=[42.7],
    )
    model: str = Field(
        description="Modelo de Groq que produjo la transcripción.",
        examples=["whisper-large-v3"],
    )
