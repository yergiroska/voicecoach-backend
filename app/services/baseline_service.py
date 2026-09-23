"""Línea base personal: las primeras sesiones del usuario frente a las últimas.

El módulo tiene dos mitades, separadas a propósito:

- Funciones PURAS (`is_valid_session`, `split_windows`, las medias, las
  tendencias y `build_baseline`). No saben nada de la base de datos: reciben
  `SessionRow` y devuelven números o schemas. Toda la lógica de la línea base
  vive aquí, y así se puede probar con listas hechas a mano.
- Una única función ASYNC, `get_baseline`, que ejecuta la consulta, convierte
  las filas y llama a `build_baseline`. No decide nada.

La consulta trae TODAS las sesiones analizadas del usuario y el filtrado se
hace en Python, no en SQL. Un usuario tendrá decenas o cientos de sesiones, y
solo se leen columnas numéricas cortas (ni el texto ni `raw_response`). A
cambio, la regla de "sesión válida" está escrita en un solo sitio y probada, en
lugar de repartida entre un `WHERE` y el código. Si algún día un usuario tiene
miles de sesiones, lo que habría que llevar a SQL es el recorte de las ventanas,
no la regla.

Reglas, todas verificadas contra datos reales antes de escribirlas:

- Una sesión es VÁLIDA si tiene al menos `MIN_WORDS` palabras y se analizó como
  español. El mínimo de palabras no es cosmético: con un audio en silencio
  Whisper se inventó un "you" en inglés, NO se marcó como `skipped` (el texto no
  estaba vacío) y el modelo devolvió `pace=5`. Sin este filtro ese dato basura
  entraría en la línea base.
- "Se analizó como español" se lee de `fillers_analyzed` y no de `language`:
  esa columna ya guarda la decisión, incluida la regla
  `ASSUME_SPANISH_WHEN_UNKNOWN` de `speech_metrics`. Volver a interpretar
  `language` aquí duplicaría el criterio y podría llegar a contradecirlo.
- De cada grabación cuenta solo su ÚLTIMO análisis, y el orden cronológico es
  el de la GRABACIÓN (`recordings.created_at`), no el del análisis: reanalizar
  un audio antiguo no lo convierte en una sesión reciente.
- Línea base = las primeras `BASELINE_SIZE` sesiones válidas. Recientes = las
  últimas `RECENT_WINDOW` POSTERIORES a la línea base. Las dos ventanas no se
  solapan nunca.
- Los scores del modelo tienen su PROPIA ventana: solo cuentan los análisis
  hechos con la versión de prompt y el modelo actuales. Las métricas
  deterministas usan todas las sesiones válidas, porque no dependen de ninguno
  de los dos.
- Los scores del modelo nunca se comparan de uno en uno: tienen un ruido medido
  de unos ±5 puntos entre pasadas. De ahí las ventanas y la zona muerta ancha
  de `SCORE_SIMILAR_BELOW`.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import fmean

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import CONNECTIVITY_ERRORS
from app.db.models import Analysis, Recording
from app.models.baseline import (
    BaselineResponse,
    BaselineStatus,
    DeterministicMetrics,
    FillerRateComparison,
    FillerTrend,
    PaceTrend,
    ScoreComparison,
    ScoresBaseline,
    ScoreTrend,
    WordsPerMinuteComparison,
)
from app.services.analysis_service import ANALYSIS_PROMPT_VERSION

# ---------------------------------------------------------------------------
# Constantes de producto. Van aquí y no en `Settings` por lo mismo que
# `ASSUME_SPANISH_WHEN_UNKNOWN`: son decisiones sobre qué significa "progreso",
# no configuración que cambie de un entorno a otro.
# ---------------------------------------------------------------------------

# Sesiones válidas que forman la línea base: las primeras N.
BASELINE_SIZE = 5

# Máximo de sesiones recientes que se comparan contra la línea base.
RECENT_WINDOW = 3

# Palabras mínimas para que una sesión cuente. Deja fuera el silencio (y las
# alucinaciones de Whisper sobre él) y las grabaciones demasiado cortas para
# decir nada del ritmo o las muletillas de alguien.
MIN_WORDS = 15

# Muestras mínimas de una métrica en cada ventana para dar su media. Por debajo
# la tendencia es `insufficient_data`: una media de un solo valor no es una
# media, sobre todo en los scores del modelo.
MIN_BASELINE_SAMPLES = 3
MIN_RECENT_SAMPLES = 2

# Tasa de muletillas (por cada 100 palabras): la diferencia se considera
# estable por debajo de max(1,0 ; 15 % de la línea base). El suelo absoluto
# evita que, con una línea base baja, pasar de 1,0 a 1,3 cuente como un cambio.
# El tramo relativo evita que, con una línea base alta, se exija bajar lo mismo
# a quien parte de 25 que a quien parte de 3.
FILLER_STABLE_ABS = 1.0
FILLER_STABLE_REL = 0.15

# Palabras por minuto: estable por debajo de ±10.
PACE_STABLE_WPM = 10.0

# Scores del modelo (0-100): por debajo de 8 puntos es `similar`; de 8 a 15,
# `better`/`worse`; desde 15, `much_better`/`much_worse`. La zona muerta supera
# con margen el ruido de ±5 de un score suelto.
SCORE_SIMILAR_BELOW = 8.0
SCORE_MUCH_FROM = 15.0

# Decimales de las cifras que se exponen (ver `_NumericComparison`).
_DECIMALS = 1


class BaselineUnavailableError(Exception):
    """No se pudo consultar la base de datos para calcular la línea base.

    Solo envuelve fallos de CONECTIVIDAD: no hay base configurada, PostgreSQL
    no responde, la conexión se cortó. El router lo traduce a 503.

    Un error de SQL o cualquier otra excepción NO se envuelve: sería un bug
    nuestro y debe salir como 500, no disfrazarse de indisponibilidad temporal.
    Mismo criterio que `get_session` en `app/core/database.py`. Es distinto del
    de `recording_repository`, que lo traduce TODO porque escribe en modo
    best-effort; esta es una lectura que sí puede fallar.
    """


@dataclass(frozen=True, slots=True)
class SessionRow:
    """Lo que la línea base necesita de una sesión: su último análisis.

    Deliberadamente sin el texto transcrito ni el `raw_response`: ni hacen
    falta, ni deben viajar por aquí.
    """

    word_count: int
    filler_count: int
    fillers_analyzed: bool
    words_per_minute: float | None
    clarity: int | None
    confidence: int | None
    pace: int | None
    prompt_version: str
    analysis_model: str


# ---------------------------------------------------------------------------
# Funciones puras
# ---------------------------------------------------------------------------


def is_valid_session(row: SessionRow) -> bool:
    """Si la sesión cuenta para la línea base (ver el docstring del módulo)."""
    return row.word_count >= MIN_WORDS and row.fillers_analyzed


def split_windows(
    rows: Sequence[SessionRow],
    *,
    baseline_size: int = BASELINE_SIZE,
    recent_window: int = RECENT_WINDOW,
) -> tuple[list[SessionRow], list[SessionRow]]:
    """Parte sesiones válidas, en orden cronológico, en (línea base, recientes).

    Las recientes salen solo de lo que queda DESPUÉS de la línea base: con 6
    sesiones hay una reciente, no tres que repitan dos de la línea base.
    Comparar una ventana consigo misma daría "estable" por construcción.
    """
    baseline = list(rows[:baseline_size])
    posteriores = rows[baseline_size:]
    recent = list(posteriores[-recent_window:]) if recent_window > 0 else []
    return baseline, recent


def filler_rate(rows: Sequence[SessionRow]) -> tuple[float | None, int]:
    """Muletillas por cada 100 palabras sobre toda la ventana, y sus muestras.

    Total entre total, no media de tasas: así una grabación de 15 palabras con
    dos muletillas (13 %) no pesa lo mismo que una de 120.
    """
    analizadas = [r for r in rows if r.fillers_analyzed and r.word_count > 0]
    palabras = sum(r.word_count for r in analizadas)
    if not analizadas or palabras == 0:
        return None, 0
    muletillas = sum(r.filler_count for r in analizadas)
    return 100.0 * muletillas / palabras, len(analizadas)


def mean_words_per_minute(rows: Sequence[SessionRow]) -> tuple[float | None, int]:
    """Media de las palabras por minuto de cada sesión, ignorando las nulas.

    Nula significa que Groq no informó de la duración: no es un ritmo de cero.
    """
    valores = [r.words_per_minute for r in rows if r.words_per_minute is not None]
    return (fmean(valores) if valores else None), len(valores)


def mean_score(
    rows: Sequence[SessionRow], score: Callable[[SessionRow], int | None]
) -> tuple[float | None, int]:
    """Media de un score del modelo, ignorando los nulos (y los nulos NO son 0)."""
    valores = [v for v in map(score, rows) if v is not None]
    return (fmean(valores) if valores else None), len(valores)


def _sufficient(baseline_samples: int, recent_samples: int) -> bool:
    return (
        baseline_samples >= MIN_BASELINE_SAMPLES
        and recent_samples >= MIN_RECENT_SAMPLES
    )


def _displayed_delta(baseline: float, recent: float) -> float:
    """La diferencia tal como la ve el cliente, redondeada.

    Sin redondear, la resta en coma flotante contradice al `delta` expuesto:
    verificado por fuerza bruta, 48 pares de ritmo con `delta` mostrado de
    +10,0 (p. ej. 54,1 -> 64,1, que da 9,999...) salían `stable` en vez de
    `faster`, y 6 de muletillas en el umbral de 1,0. Solo se aplica a las
    métricas deterministas: las medias de los scores no se exponen, y en ellas
    no apareció ningún caso en 860 fronteras exactas probadas.
    """
    return round(recent - baseline, _DECIMALS)


def filler_trend(baseline: float | None, recent: float | None) -> FillerTrend:
    """Tendencia de la tasa de muletillas. Bajar es mejorar.

    El umbral es estricto: una diferencia exactamente igual al umbral ya cuenta
    como cambio.
    """
    if baseline is None or recent is None:
        return FillerTrend.INSUFFICIENT_DATA
    delta = _displayed_delta(baseline, recent)
    # El umbral también se redondea (a más decimales que el delta) para que un
    # 15 % que debería dar 3,0 exacto no llegue como 3,0000000000000004.
    umbral = round(max(FILLER_STABLE_ABS, FILLER_STABLE_REL * baseline), 6)
    if abs(delta) < umbral:
        return FillerTrend.STABLE
    return FillerTrend.IMPROVING if delta < 0 else FillerTrend.WORSENING


def pace_trend(baseline: float | None, recent: float | None) -> PaceTrend:
    """Tendencia de las palabras por minuto. Sin juicio de valor (ver `PaceTrend`)."""
    if baseline is None or recent is None:
        return PaceTrend.INSUFFICIENT_DATA
    delta = _displayed_delta(baseline, recent)
    if abs(delta) < PACE_STABLE_WPM:
        return PaceTrend.STABLE
    return PaceTrend.FASTER if delta > 0 else PaceTrend.SLOWER


def score_trend(baseline: float | None, recent: float | None) -> ScoreTrend:
    """Tendencia cualitativa de un score del modelo. Subir es mejorar."""
    if baseline is None or recent is None:
        return ScoreTrend.INSUFFICIENT_DATA
    delta = recent - baseline
    magnitud = abs(delta)
    if magnitud < SCORE_SIMILAR_BELOW:
        return ScoreTrend.SIMILAR
    if magnitud < SCORE_MUCH_FROM:
        return ScoreTrend.BETTER if delta > 0 else ScoreTrend.WORSE
    return ScoreTrend.MUCH_BETTER if delta > 0 else ScoreTrend.MUCH_WORSE


def _numeric_values(
    baseline_value: float | None,
    baseline_samples: int,
    recent_value: float | None,
    recent_samples: int,
) -> tuple[float | None, float | None, float | None]:
    """Aplica los mínimos de muestra y redondea. Devuelve (base, recientes, delta).

    La tendencia se calcula sobre estas cifras YA redondeadas, igual que el
    `delta`: así lo que ve el cliente es coherente consigo mismo (`delta` es
    exactamente `recent - baseline`, y la tendencia es la que corresponde a esos
    números, no a unos decimales ocultos).
    """
    base = (
        round(baseline_value, _DECIMALS)
        if baseline_value is not None and baseline_samples >= MIN_BASELINE_SAMPLES
        else None
    )
    recientes = (
        round(recent_value, _DECIMALS)
        if recent_value is not None and recent_samples >= MIN_RECENT_SAMPLES
        else None
    )
    delta = (
        round(recientes - base, _DECIMALS)
        if base is not None and recientes is not None
        else None
    )
    return base, recientes, delta


def _compare_metrics(
    baseline: Sequence[SessionRow], recent: Sequence[SessionRow]
) -> DeterministicMetrics:
    base_fr, n_base_fr = filler_rate(baseline)
    rec_fr, n_rec_fr = filler_rate(recent)
    fr_base, fr_rec, fr_delta = _numeric_values(base_fr, n_base_fr, rec_fr, n_rec_fr)

    base_wpm, n_base_wpm = mean_words_per_minute(baseline)
    rec_wpm, n_rec_wpm = mean_words_per_minute(recent)
    wpm_base, wpm_rec, wpm_delta = _numeric_values(
        base_wpm, n_base_wpm, rec_wpm, n_rec_wpm
    )

    return DeterministicMetrics(
        filler_rate=FillerRateComparison(
            baseline=fr_base,
            recent=fr_rec,
            delta=fr_delta,
            trend=filler_trend(fr_base, fr_rec),
            baseline_samples=n_base_fr,
            recent_samples=n_rec_fr,
        ),
        words_per_minute=WordsPerMinuteComparison(
            baseline=wpm_base,
            recent=wpm_rec,
            delta=wpm_delta,
            trend=pace_trend(wpm_base, wpm_rec),
            baseline_samples=n_base_wpm,
            recent_samples=n_rec_wpm,
        ),
    )


def _compare_score(
    baseline: Sequence[SessionRow],
    recent: Sequence[SessionRow],
    score: Callable[[SessionRow], int | None],
) -> ScoreComparison:
    base, n_base = mean_score(baseline, score)
    rec, n_rec = mean_score(recent, score)
    # Sin redondear: estas medias no se exponen, solo deciden la tendencia.
    if not _sufficient(n_base, n_rec):
        trend = ScoreTrend.INSUFFICIENT_DATA
    else:
        trend = score_trend(base, rec)
    return ScoreComparison(trend=trend, baseline_samples=n_base, recent_samples=n_rec)


def _build_scores(
    valid: Sequence[SessionRow], *, prompt_version: str, analysis_model: str
) -> ScoresBaseline:
    comparables = [
        r
        for r in valid
        if r.prompt_version == prompt_version and r.analysis_model == analysis_model
    ]
    needed = max(0, BASELINE_SIZE - len(comparables))
    common = {
        "prompt_version": prompt_version,
        "analysis_model": analysis_model,
        "valid_sessions": len(comparables),
        "sessions_needed": needed,
    }
    if needed:
        return ScoresBaseline(status=BaselineStatus.COLLECTING, **common)

    baseline, recent = split_windows(comparables)
    return ScoresBaseline(
        status=BaselineStatus.READY,
        clarity=_compare_score(baseline, recent, lambda r: r.clarity),
        confidence=_compare_score(baseline, recent, lambda r: r.confidence),
        pace=_compare_score(baseline, recent, lambda r: r.pace),
        **common,
    )


def build_baseline(
    rows: Sequence[SessionRow], *, prompt_version: str, analysis_model: str
) -> BaselineResponse:
    """Calcula la respuesta completa a partir de las sesiones del usuario.

    Args:
        rows: TODAS las sesiones analizadas del usuario (válidas o no), una por
            grabación y en orden cronológico de grabación.
        prompt_version: versión de prompt actual; solo esos scores cuentan.
        analysis_model: modelo actual; solo esos scores cuentan.
    """
    valid = [r for r in rows if is_valid_session(r)]
    needed = max(0, BASELINE_SIZE - len(valid))

    metrics = None
    if not needed:
        baseline, recent = split_windows(valid)
        metrics = _compare_metrics(baseline, recent)

    return BaselineResponse(
        status=BaselineStatus.COLLECTING if needed else BaselineStatus.READY,
        baseline_size=BASELINE_SIZE,
        recent_window=RECENT_WINDOW,
        valid_sessions=len(valid),
        excluded_sessions=len(rows) - len(valid),
        sessions_needed=needed,
        metrics=metrics,
        scores=_build_scores(
            valid, prompt_version=prompt_version, analysis_model=analysis_model
        ),
    )


# ---------------------------------------------------------------------------
# Acceso a la base de datos
# ---------------------------------------------------------------------------


def _motivo(exc: BaseException) -> str:
    """Primera línea del error: la parte accionable, sin el `DETAIL:` de PostgreSQL.

    Mismo criterio que `recording_repository._motivo`. En una lectura el DETAIL
    no debería llevar datos del usuario, pero no cuesta nada no arriesgarse.
    """
    texto = str(exc)
    return texto.splitlines()[0] if texto else type(exc).__name__


async def fetch_sessions(session: AsyncSession, uid: str) -> list[SessionRow]:
    """Último análisis de cada grabación del usuario, en orden cronológico.

    Una sola consulta: `DISTINCT ON (recording_id)` con el análisis más reciente
    primero se queda con uno por grabación, y la consulta exterior ordena por la
    fecha de la GRABACIÓN. El id de la grabación desempata dos sesiones con el
    mismo `created_at` para que el orden sea estable entre llamadas.

    Filtra por `analyses.user_uid`, la columna denormalizada que existe
    justamente para esto, y la cubre `ix_analyses_user_uid_created_at`.

    Raises:
        BaselineUnavailableError: fallo de conectividad con la base de datos.
    """
    ultimo = (
        select(
            Analysis.word_count,
            Analysis.filler_count,
            Analysis.fillers_analyzed,
            Analysis.words_per_minute,
            Analysis.clarity_score,
            Analysis.confidence_score,
            Analysis.pace_score,
            Analysis.prompt_version,
            Analysis.analysis_model,
            Recording.id.label("recording_id"),
            Recording.created_at.label("session_at"),
        )
        .join(Recording, Recording.id == Analysis.recording_id)
        .where(Analysis.user_uid == uid)
        .distinct(Analysis.recording_id)
        .order_by(Analysis.recording_id, Analysis.created_at.desc())
        .subquery()
    )
    statement = select(ultimo).order_by(ultimo.c.session_at, ultimo.c.recording_id)

    # La traducción se hace aquí y no se deja a `get_session`: la conexión es
    # perezosa y el fallo salta en este `execute`. Con un error de dominio
    # propio, el router decide el 503 sin depender de cómo propaga FastAPI una
    # excepción a través del `yield` de la dependency.
    try:
        result = await session.execute(statement)
    except CONNECTIVITY_ERRORS as exc:
        raise BaselineUnavailableError(_motivo(exc)) from exc

    return [
        SessionRow(
            word_count=fila.word_count,
            filler_count=fila.filler_count,
            fillers_analyzed=fila.fillers_analyzed,
            words_per_minute=fila.words_per_minute,
            clarity=fila.clarity_score,
            confidence=fila.confidence_score,
            pace=fila.pace_score,
            prompt_version=fila.prompt_version,
            analysis_model=fila.analysis_model,
        )
        for fila in result
    ]


async def get_baseline(session: AsyncSession, uid: str) -> BaselineResponse:
    """Línea base del usuario con la versión de prompt y el modelo actuales.

    Raises:
        BaselineUnavailableError: fallo de conectividad con la base de datos.
    """
    rows = await fetch_sessions(session, uid)
    return build_baseline(
        rows,
        prompt_version=ANALYSIS_PROMPT_VERSION,
        analysis_model=settings.groq_analysis_model,
    )
