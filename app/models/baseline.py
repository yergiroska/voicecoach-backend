"""Schemas Pydantic de la línea base personal (`GET /me/baseline`).

La línea base es el promedio de las primeras sesiones VÁLIDAS del usuario, y la
respuesta compara contra ella sus sesiones más recientes. Una sesión es válida
si tiene al menos un mínimo de palabras y se analizó como español; el silencio
(en el que Whisper llega a inventarse texto, verificado: devolvió "you" en
inglés), los audios muy cortos y los de otro idioma no cuentan.

Las dos familias de datos se exponen de forma DISTINTA a propósito:

- Las métricas deterministas (`metrics`) llevan números: la tasa de muletillas
  y las palabras por minuto son aritmética exacta sobre el texto, y un "de 23,4
  a 0,0 muletillas por cada 100 palabras" es un dato real.
- Los scores del modelo (`scores`) llevan SOLO una tendencia en palabras, sin
  números. Tienen un ruido medido de unos ±5 puntos entre pasadas incluso con
  `temperature=0`, así que un "has pasado de 65 a 60" le estaría reportando
  ruido al usuario. Los valores siguen guardados en `analyses` para quien los
  necesite; lo que no se hace es presentarlos como si fueran precisos.

Por la misma razón las tendencias nunca comparan una sesión suelta: comparan la
media de una ventana de sesiones recientes contra la media de la línea base.

Los umbrales, los tamaños de ventana y los mínimos de muestra viven en
`app/services/baseline_service.py`, no aquí: este módulo describe la forma de
la respuesta, no las reglas que la rellenan.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class BaselineStatus(StrEnum):
    """Si ya hay sesiones válidas suficientes para fijar la línea base."""

    COLLECTING = "collecting"
    READY = "ready"


class FillerTrend(StrEnum):
    """Evolución de la tasa de muletillas. Menos es mejor."""

    IMPROVING = "improving"
    STABLE = "stable"
    WORSENING = "worsening"
    INSUFFICIENT_DATA = "insufficient_data"


class PaceTrend(StrEnum):
    """Evolución de las palabras por minuto.

    Sin "mejor" ni "peor" a propósito: el ritmo no mejora al subir ni al bajar,
    tiene un rango adecuado (orientativamente, de 110 a 170 ppm en español). Si
    esto dijera "improving" al acelerar, animaría a atropellarse.
    """

    FASTER = "faster"
    STABLE = "stable"
    SLOWER = "slower"
    INSUFFICIENT_DATA = "insufficient_data"


class ScoreTrend(StrEnum):
    """Evolución cualitativa de un score del modelo.

    Cinco tramos y no tres: con la zona muerta ancha que exige el ruido del
    modelo, "better" y "much_better" permiten distinguir una mejora dudosa de
    una evidente sin tener que enseñar ningún número.
    """

    MUCH_BETTER = "much_better"
    BETTER = "better"
    SIMILAR = "similar"
    WORSE = "worse"
    MUCH_WORSE = "much_worse"
    INSUFFICIENT_DATA = "insufficient_data"


# `insufficient_data` es un valor del enum y no un `null` por la misma regla que
# sigue todo el proyecto: `null` no debe tener que interpretarse. Con un valor
# explícito el cliente sabe que falta muestra y puede decirlo, en vez de
# adivinar si "sin tendencia" significa "estable" o "no se sabe".
_INSUFFICIENT_DATA_NOTE = (
    "`insufficient_data` cuando alguna de las dos ventanas no llega al mínimo "
    "de muestras de esta métrica: no es «estable», es «todavía no se sabe»."
)


class _WindowSamples(BaseModel):
    """Cuántas sesiones aportaron un valor a cada ventana.

    Se exponen siempre porque una media no dice nada sin su tamaño de muestra,
    y porque cada métrica puede tener un número distinto: una sesión sin
    duración de audio no tiene palabras por minuto pero sí muletillas, y una en
    la que falló el modelo no tiene scores pero sí métricas.
    """

    baseline_samples: int = Field(
        ge=0,
        description="Sesiones de la línea base con un valor para esta métrica.",
        examples=[5],
    )
    recent_samples: int = Field(
        ge=0,
        description=(
            "Sesiones recientes con un valor para esta métrica. Es 0 mientras el "
            "usuario no tenga sesiones posteriores a la línea base: las dos "
            "ventanas nunca se solapan."
        ),
        examples=[3],
    )


class _NumericComparison(_WindowSamples):
    """Línea base frente a recientes para una métrica determinista.

    Las cifras llegan redondeadas a un decimal: la precisión de los datos de
    origen (la duración que informa Whisper, en particular) no da para más.
    """

    baseline: float | None = Field(
        default=None,
        description=(
            "Media de la línea base. `null` si no alcanza el mínimo de muestras."
        ),
    )
    recent: float | None = Field(
        default=None,
        description=(
            "Media de las sesiones recientes. `null` si no alcanza el mínimo de "
            "muestras."
        ),
    )
    delta: float | None = Field(
        default=None,
        description=(
            "`recent - baseline`. `null` si falta cualquiera de los dos. El "
            "signo por sí solo no dice si es bueno o malo: para eso está `trend`."
        ),
    )


class FillerRateComparison(_NumericComparison):
    """Tasa de muletillas: cuántas por cada 100 palabras.

    Una tasa y no un conteo porque las sesiones tienen longitudes distintas: una
    grabación de un minuto tendrá más muletillas que una de quince segundos sin
    que eso signifique nada. Se calcula en conjunto sobre la ventana (total de
    muletillas entre total de palabras), así que las sesiones largas pesan más
    que las cortas, que es lo que se quiere.
    """

    trend: FillerTrend = Field(
        description=f"Hacia dónde evoluciona. {_INSUFFICIENT_DATA_NOTE}",
        examples=[FillerTrend.IMPROVING],
    )


class WordsPerMinuteComparison(_NumericComparison):
    """Palabras por minuto: media de las de cada sesión de la ventana."""

    trend: PaceTrend = Field(
        description=f"Hacia dónde evoluciona. {_INSUFFICIENT_DATA_NOTE}",
        examples=[PaceTrend.STABLE],
    )


class DeterministicMetrics(BaseModel):
    """Métricas calculadas en Python, independientes del prompt y del modelo.

    Por eso su ventana usa todas las sesiones válidas, sea cual sea la versión
    del prompt con la que se analizaron.
    """

    filler_rate: FillerRateComparison = Field(
        description="Muletillas por cada 100 palabras.",
        examples=[
            {
                "baseline": 23.4,
                "recent": 0.0,
                "delta": -23.4,
                "trend": "improving",
                "baseline_samples": 5,
                "recent_samples": 3,
            }
        ],
    )
    words_per_minute: WordsPerMinuteComparison = Field(
        description="Ritmo del habla.",
        examples=[
            {
                "baseline": 141.7,
                "recent": 146.7,
                "delta": 5.0,
                "trend": "stable",
                "baseline_samples": 5,
                "recent_samples": 3,
            }
        ],
    )


class ScoreComparison(_WindowSamples):
    """Tendencia de un score del modelo. Sin cifras, a propósito.

    Ver el docstring del módulo: con ±5 puntos de ruido entre pasadas, exponer
    las medias invitaría a leer como progreso lo que es variación del modelo.
    """

    trend: ScoreTrend = Field(
        description=f"Tendencia cualitativa. {_INSUFFICIENT_DATA_NOTE}",
        examples=[ScoreTrend.MUCH_BETTER],
    )


class ScoresBaseline(BaseModel):
    """Línea base de los scores del modelo, con su PROPIA ventana.

    Solo cuentan los análisis hechos con la versión de prompt y el modelo
    ACTUALES: dos scores de prompts o modelos distintos no son comparables, y
    mezclarlos haría pasar un cambio nuestro por un progreso del usuario. La
    consecuencia es que, al cambiar el prompt o el modelo, esta sección vuelve a
    `collecting` y se reconstruye con datos nuevos, mientras que `metrics`
    sigue lista porque no depende de ninguno de los dos.
    """

    status: BaselineStatus = Field(
        description=(
            "Estado de ESTA ventana, independiente del `status` general: puede "
            "estar en `collecting` con las métricas ya listas."
        ),
        examples=[BaselineStatus.READY],
    )
    prompt_version: str = Field(
        description="Versión de prompt con la que se calcula esta sección.",
        examples=["v1"],
    )
    analysis_model: str = Field(
        description="Modelo de Groq con el que se calcula esta sección.",
        examples=["openai/gpt-oss-120b"],
    )
    valid_sessions: int = Field(
        ge=0,
        description=(
            "Sesiones válidas analizadas con este prompt y este modelo. Puede "
            "ser menor que el `valid_sessions` general."
        ),
        examples=[9],
    )
    sessions_needed: int = Field(
        ge=0,
        description="Sesiones válidas que faltan para `ready`. 0 si ya lo está.",
        examples=[0],
    )
    clarity: ScoreComparison | None = Field(
        default=None,
        description="Claridad. `null` mientras `status` sea `collecting`.",
    )
    confidence: ScoreComparison | None = Field(
        default=None,
        description="Confianza. `null` mientras `status` sea `collecting`.",
    )
    pace: ScoreComparison | None = Field(
        default=None,
        description=(
            "Adecuación del ritmo según el modelo. No confundir con "
            "`metrics.words_per_minute`, que es la medida objetiva. `null` "
            "mientras `status` sea `collecting`."
        ),
    )


class BaselineResponse(BaseModel):
    """Línea base personal del usuario y cómo evolucionan sus sesiones recientes.

    Mientras `status` sea `collecting`, `metrics` es `null` y el cliente debería
    mostrar el progreso (`valid_sessions` de `baseline_size`) en vez de una
    comparación. Con `ready` y aún sin sesiones posteriores a la línea base, las
    comparaciones vienen con `trend: insufficient_data` y `recent_samples: 0`.
    """

    status: BaselineStatus = Field(
        description=(
            "`ready` cuando hay al menos `baseline_size` sesiones válidas. "
            "Gobierna `metrics`; los scores tienen su propio estado en "
            "`scores.status`."
        ),
        examples=[BaselineStatus.READY],
    )
    baseline_size: int = Field(
        ge=1,
        description="Sesiones válidas que forman la línea base: las primeras N.",
        examples=[5],
    )
    recent_window: int = Field(
        ge=1,
        description=(
            "Máximo de sesiones recientes (las últimas, siempre posteriores a la "
            "línea base) que se comparan contra ella."
        ),
        examples=[3],
    )
    valid_sessions: int = Field(
        ge=0,
        description="Sesiones analizadas que cuentan para la línea base.",
        examples=[9],
    )
    excluded_sessions: int = Field(
        ge=0,
        description=(
            "Sesiones analizadas que NO cuentan: silencio, menos del mínimo de "
            "palabras, u otro idioma. Se expone para que el cliente pueda "
            "explicar por qué una grabación no ha hecho avanzar el progreso. "
            "Las grabaciones sin análisis guardado no aparecen en ningún "
            "contador: sin métricas no hay nada que comparar."
        ),
        examples=[3],
    )
    sessions_needed: int = Field(
        ge=0,
        description="Sesiones válidas que faltan para `ready`. 0 si ya lo está.",
        examples=[0],
    )
    metrics: DeterministicMetrics | None = Field(
        default=None,
        description=(
            "Comparación de las métricas deterministas. `null` mientras "
            "`status` sea `collecting`."
        ),
    )
    scores: ScoresBaseline = Field(
        description=(
            "Tendencia de los scores del modelo. Siempre presente, con su propio "
            "`status`."
        ),
    )
