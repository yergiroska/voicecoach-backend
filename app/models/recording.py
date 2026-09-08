"""Schemas Pydantic de las grabaciones de audio."""

from pydantic import BaseModel, Field

from app.models.analysis import RecordingAnalysis


class RecordingUploadResponse(BaseModel):
    """Respuesta de `POST /recordings`: el audio recibido y ya transcrito.

    La transcripción es síncrona, así que llega en esta misma respuesta. El
    archivo de audio se borra del disco en cuanto la grabación queda guardada en
    la base de datos, de modo que los campos `filename`/`size_bytes` describen
    algo que ya no existe cuando el cliente lee esto: se mantienen porque le
    sirven para verificar que llegó lo que envió.

    Incluye el análisis de la comunicación en `analysis`, que puede venir a
    `null`: la transcripción y el análisis se obtienen por caminos distintos y
    el segundo es opcional (ver la descripción de ese campo).
    """

    recording_id: str = Field(
        description=(
            "Identificador único de la grabación, y clave primaria de su fila "
            "en la base de datos. Es el mismo identificador que aparece en los "
            "logs del servidor. Excepción: si el guardado falló, la "
            "transcripción se devuelve igualmente (201) y este id no "
            "corresponde a ninguna fila."
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
    analysis: RecordingAnalysis | None = Field(
        default=None,
        description=(
            "Análisis de la comunicación construido sobre `text`.\n\n"
            "PUEDE SER `null`, y el cliente tiene que estar preparado para eso. "
            "El análisis se genera después de la transcripción y de forma "
            "independiente: si el modelo de análisis no responde, agota su "
            "cuota o devuelve algo inservible, la respuesta sigue siendo 201 "
            "con la transcripción, que es lo que no se puede recuperar. Un "
            "cliente que dé por hecho este objeto se romperá el día que Groq "
            "tenga una mala tarde.\n\n"
            "El análisis tampoco se pide cuando la transcripción no tiene "
            "palabras (audio en silencio): en ese caso `analysis` sí llega, "
            "pero con las métricas a cero y los tres scores a `null`.\n\n"
            "Que este campo venga con datos no garantiza que se haya guardado "
            "en la base: la escritura también es independiente. Se devuelve lo "
            "que se calculó, aunque el guardado haya fallado."
        ),
    )
