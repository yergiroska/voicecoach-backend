"""Schemas Pydantic del análisis de comunicación.

Hay DOS contratos aquí, y están separados a propósito:

- `RecordingAnalysis` y sus piezas — lo que el backend devuelve al móvil. Es
  contrato público: cambiarlo rompe clientes.
- `AnalysisLlmOutput` — lo que se le pide al modelo de lenguaje y con lo que se
  valida su respuesta. Es un detalle interno que cambiará con cada versión del
  prompt.

Reutilizar un solo schema para ambas cosas sería el error fácil: acoplaría la
API pública a la forma del prompt, de modo que afinar el prompt obligaría a
versionar la API. Por eso el servicio de análisis traduce de uno a otro.

La otra separación importante es de dónde sale cada número:

- `SpeechMetrics` son métricas DETERMINISTAS, calculadas en Python. Contar
  palabras, calcular palabras por minuto y detectar muletillas es aritmética:
  un LLM lo hace peor, cuesta tokens y no da el mismo resultado dos veces.
- `CommunicationScores` son JUICIOS del modelo. Claridad, confianza y ritmo sí
  son valoraciones subjetivas, y ahí el LLM aporta algo que `len(texto.split())`
  no puede.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Límites de los scores del modelo. Fuera de este rango el valor se descarta
# (ver `_score_valido`): un 120 en claridad no es un dato, es un modelo
# alucinando, y guardarlo contaminaría la línea base de la Fase 7.
SCORE_MIN = 0
SCORE_MAX = 100

# Tope de sugerencias que se aceptan del modelo. No es una restricción de
# producto, es un cortafuegos: si el prompt sale mal y el modelo devuelve
# doscientas sugerencias, ni se guardan ni se mandan al móvil.
MAX_SUGGESTIONS = 10

# Tope del resumen, en caracteres. Mismo motivo. Se recorta, no se rechaza: un
# resumen demasiado largo sigue siendo útil truncado, y tirar el análisis
# completo por eso sería peor.
MAX_SUMMARY_CHARS = 1000


def _score_valido(value: object) -> int | None:
    """Normaliza un score del modelo, o devuelve `None` si no sirve.

    Tolerante a propósito. El modelo puede devolver `85`, `85.0`, `"85"`, `null`,
    `"alta"` o `120`, y ninguna de esas variantes debe tumbar el análisis
    completo: las métricas deterministas que lo acompañan son válidas de todos
    modos, y descartarlas por un score raro sería perder lo bueno por lo malo.

    Devolver `None` en vez de recortar a los límites es deliberado: un 120
    recortado a 100 se guardaría como si el modelo hubiera dicho 100, y esa
    invención luego no se distingue de un dato real. `None` es honesto.
    """
    if value is None:
        return None

    if isinstance(value, bool):
        # `bool` es subclase de `int` en Python: sin este caso, un `true` del
        # modelo se guardaría como un score de 1.
        return None

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None

    try:
        numero = round(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

    if not SCORE_MIN <= numero <= SCORE_MAX:
        return None

    return numero


class FillerWord(BaseModel):
    """Una muletilla detectada y cuántas veces aparece."""

    word: str = Field(
        description="La muletilla, normalizada a minúsculas.",
        examples=["eh"],
    )
    count: int = Field(
        ge=1,
        description="Veces que aparece en la transcripción.",
        examples=[7],
    )


class SpeechMetrics(BaseModel):
    """Métricas objetivas de la transcripción, calculadas en Python.

    Ninguna de estas la produce el modelo de lenguaje. Están siempre presentes
    (salvo `words_per_minute`, que necesita la duración del audio, y las
    muletillas, que necesitan que el idioma sea español): son las que hacen que
    el análisis siga aportando algo incluso si la llamada al LLM falla.
    """

    word_count: int = Field(
        ge=0,
        description=(
            "Palabras de la transcripción. Es 0 si el audio era silencio, caso "
            "válido y ya documentado en `RecordingUploadResponse.text`."
        ),
        examples=[38],
    )
    words_per_minute: float | None = Field(
        default=None,
        description=(
            "Ritmo del habla. Es `null` cuando Groq no informó de la duración "
            "del audio, porque sin ella no se puede calcular: se deja vacío en "
            "vez de estimarlo. Referencia orientativa en español: por debajo de "
            "unas 110 ppm se percibe lento y por encima de unas 170, atropellado."
        ),
        examples=[120.1],
    )
    filler_count: int = Field(
        ge=0,
        description="Total de muletillas, sumando todas las de `filler_words`.",
        examples=[6],
    )
    filler_words: list[FillerWord] = Field(
        default_factory=list,
        description=(
            "Desglose por muletilla, de la más frecuente a la menos. Lista vacía "
            "si no se detectó ninguna. Ver `fillers_analyzed` antes de "
            "interpretar una lista vacía."
        ),
        examples=[[{"word": "eh", "count": 3}, {"word": "o sea", "count": 3}]],
    )
    fillers_analyzed: bool = Field(
        default=True,
        description=(
            "Si la detección de muletillas se ejecutó. Es `false` cuando el "
            "idioma de la grabación no es español: la lista de muletillas es "
            "es-ES y aplicarla a otro idioma daría un resultado sin sentido.\n\n"
            "DISTINGUIR ESTO IMPORTA. Con `false`, `filler_count=0` y "
            "`filler_words=[]` significan «no se analizó», NO «no se encontró "
            "ninguna»: una interfaz que muestre «0 muletillas, ¡perfecto!» en "
            "un audio en inglés estaría mintiendo. Con `false` hay que ocultar "
            "el apartado de muletillas, no enseñarlo a cero.\n\n"
            "`word_count` y `words_per_minute` no dependen del idioma y siguen "
            "siendo válidos en ambos casos."
        ),
        examples=[True],
    )


class CommunicationScores(BaseModel):
    """Valoraciones del modelo, de 0 a 100.

    Los tres campos son opcionales y hay que tratarlos como tales en el cliente:
    `null` significa "el modelo no dio un valor usable para esto", no cero. Un
    0 es una valoración muy mala; `null` es la ausencia de valoración, y
    confundirlos en la interfaz sería un error visible para el usuario.
    """

    clarity: int | None = Field(
        default=None,
        ge=SCORE_MIN,
        le=SCORE_MAX,
        description="Claridad del mensaje: si la idea se entiende sin esfuerzo.",
        examples=[62],
    )
    confidence: int | None = Field(
        default=None,
        ge=SCORE_MIN,
        le=SCORE_MAX,
        description=(
            "Seguridad que transmite el discurso: afirmaciones frente a titubeos, "
            "rodeos y disculpas innecesarias."
        ),
        examples=[55],
    )
    pace: int | None = Field(
        default=None,
        ge=SCORE_MIN,
        le=SCORE_MAX,
        description=(
            "Adecuación del ritmo. El modelo lo valora a partir de las "
            "`words_per_minute` YA CALCULADAS, no las estima: aquí solo juzga si "
            "ese ritmo acompaña o estorba al mensaje."
        ),
        examples=[70],
    )


class RecordingAnalysis(BaseModel):
    """El análisis de una grabación, tal como lo recibe el móvil.

    Es el objeto que en el paso 14 se anidará en `RecordingUploadResponse` como
    campo `analysis`, opcional: si el análisis falla, la respuesta sigue siendo
    201 con la transcripción y `analysis: null`. Un cliente que asuma que este
    objeto viene siempre está mal escrito.

    Deliberadamente NO trae el id de la fila de `analyses` ni su `created_at`:
    este objeto se construye ANTES de guardarlo (y el guardado es best-effort,
    puede no ocurrir), así que exponer campos que solo existen si la escritura
    salió bien obligaría a que fuesen opcionales por un motivo que al cliente no
    le dice nada. Se añadirán cuando haya un endpoint que lea análisis ya
    guardados.
    """

    metrics: SpeechMetrics = Field(
        description="Métricas objetivas calculadas sobre el texto."
    )
    scores: CommunicationScores = Field(
        description="Valoraciones del modelo, de 0 a 100."
    )
    summary: str | None = Field(
        default=None,
        description=(
            "Devolución breve en prosa, en segunda persona y en el idioma de la "
            "grabación. `null` si el modelo no la produjo."
        ),
        examples=["Se te entiende bien, pero las muletillas te frenan el ritmo."],
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description=(
            "Consejos concretos y accionables, ordenados por impacto. Lista "
            "vacía si el modelo no produjo ninguno."
        ),
        examples=[["Sustituye 'o sea' por una pausa breve.", "Cierra cada idea antes de empezar la siguiente."]],
    )
    model: str = Field(
        description=(
            "Modelo de Groq que produjo la valoración. Se corresponde con la "
            "columna `analyses.analysis_model`. Ojo: no es el mismo que el "
            "`model` del nivel superior de la respuesta, que es el de Whisper."
        ),
        examples=["llama-3.3-70b-versatile"],
    )
    prompt_version: str = Field(
        description=(
            "Versión del prompt usada. Se expone —y no solo se guarda— para que "
            "el cliente pueda distinguir dos análisis del mismo audio hechos con "
            "prompts distintos, en vez de leerlos como una mejora del usuario."
        ),
        examples=["v1"],
    )


class AnalysisLlmOutput(BaseModel):
    """Lo que se le pide al modelo, y con lo que se valida su respuesta.

    CONTRATO INTERNO, no público: `RecordingAnalysis` es lo que ve el móvil.

    Esta clase hace doble trabajo: su `model_json_schema()` es lo que se le pasa
    a Groq como esquema de salida estructurada, y la propia clase valida lo que
    vuelve. Así el prompt y el parseo no se pueden desincronizar.

    Solo pide lo que un LLM debe decidir. Ni `word_count`, ni
    `words_per_minute`, ni el conteo de muletillas: eso lo calcula Python en el
    paso 10, y de hecho las `words_per_minute` se le dan al modelo YA hechas
    para que las interprete.

    Todo es opcional y todos los validadores perdonan, con una regla: una
    respuesta a medias es mejor que ninguna. Las métricas deterministas que
    acompañan al análisis son válidas de todos modos, así que un score
    inventado o un campo que falta degradan el resultado en vez de tirarlo.
    """

    # `extra="ignore"`: los modelos añaden campos que no se pidieron. No es
    # motivo para fallar, y nada se pierde: la respuesta cruda se guarda
    # completa en `analyses.raw_response` (JSONB), de donde se puede extraer
    # después sin volver a pagar el análisis.
    model_config = ConfigDict(extra="ignore")

    # Los `ge`/`le` están aquí para que aparezcan en `model_json_schema()`: es
    # así como el modelo se entera de que la escala es 0-100. Son compatibles
    # con los validadores tolerantes de abajo porque esos corren en
    # `mode="before"`, o sea ANTES de comprobar los límites: un 120 ya llega
    # convertido en `None` y la restricción nunca lo ve. Sin `mode="before"`
    # esta combinación haría fallar la validación en vez de degradarla.
    clarity_score: int | None = Field(default=None, ge=SCORE_MIN, le=SCORE_MAX)
    confidence_score: int | None = Field(default=None, ge=SCORE_MIN, le=SCORE_MAX)
    pace_score: int | None = Field(default=None, ge=SCORE_MIN, le=SCORE_MAX)
    summary: str | None = Field(default=None, max_length=MAX_SUMMARY_CHARS)
    suggestions: list[str] = Field(default_factory=list, max_length=MAX_SUGGESTIONS)

    @field_validator("clarity_score", "confidence_score", "pace_score", mode="before")
    @classmethod
    def _normalizar_score(cls, value: object) -> int | None:
        return _score_valido(value)

    @field_validator("summary", mode="before")
    @classmethod
    def _normalizar_summary(cls, value: object) -> str | None:
        if not isinstance(value, str):
            # Ni un número ni un objeto son un resumen. Se descarta en silencio
            # en vez de intentar convertirlo: el texto crudo sigue en
            # `raw_response` si hiciera falta rescatarlo.
            return None
        texto = value.strip()
        if not texto:
            return None
        return texto[:MAX_SUMMARY_CHARS]

    @field_validator("suggestions", mode="before")
    @classmethod
    def _normalizar_suggestions(cls, value: object) -> list[str]:
        if value is None:
            return []
        # Un modelo puede devolver una sola sugerencia como string en vez de
        # como lista de un elemento.
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []

        limpias = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        return limpias[:MAX_SUGGESTIONS]

    def to_scores(self) -> CommunicationScores:
        """Traduce los scores al schema público.

        Este método ES la frontera entre el contrato interno y el público, y
        existe para que el renombrado (`clarity_score` -> `clarity`) esté en un
        solo sitio: si el prompt cambia los nombres, se toca aquí y la API
        pública no se mueve.
        """
        return CommunicationScores(
            clarity=self.clarity_score,
            confidence=self.confidence_score,
            pace=self.pace_score,
        )
