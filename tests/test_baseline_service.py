"""Tests de las funciones puras de `app/services/baseline_service.py`.

Ninguno toca la base de datos ni la red: la lógica de la línea base vive en
funciones que reciben `SessionRow` y devuelven números o schemas, precisamente
para poder probarla así.

`REAL_SESSIONS` son las 12 sesiones de la base de desarrollo tal como las
devolvió `fetch_sessions` al verificar el Bloque 7.1, congeladas aquí. Incluyen
los casos límite que aparecieron de verdad: una sesión de 10 palabras, otra en
inglés y el silencio en el que Whisper se inventó un "you".

Lo que NO está aquí, y se verificó a mano contra PostgreSQL real: la consulta
de `fetch_sessions` y la traducción de los fallos de conexión a
`BaselineUnavailableError` / 503. Aquí solo hay un test de regresión que
comprueba que las excepciones que se vieron siguen en `CONNECTIVITY_ERRORS`.
"""

import socket

import pytest
from asyncpg.exceptions import ConnectionDoesNotExistError

from app.core.database import CONNECTIVITY_ERRORS
from app.models.baseline import (
    BaselineStatus,
    FillerTrend,
    PaceTrend,
    ScoreTrend,
)
from app.services import baseline_service as bs
from app.services.baseline_service import SessionRow, build_baseline

MODEL = "openai/gpt-oss-120b"
PROMPT = "v1"


def row(
    *,
    words: int = 50,
    fillers: int = 0,
    wpm: float | None = 140.0,
    clarity: int | None = 70,
    confidence: int | None = 70,
    pace: int | None = 70,
    analyzed: bool = True,
    prompt_version: str = PROMPT,
    model: str = MODEL,
) -> SessionRow:
    """Una sesión válida y anodina por defecto; cada test cambia solo lo suyo."""
    return SessionRow(
        word_count=words,
        filler_count=fillers,
        fillers_analyzed=analyzed,
        words_per_minute=wpm,
        clarity=clarity,
        confidence=confidence,
        pace=pace,
        prompt_version=prompt_version,
        analysis_model=model,
    )


def _real(words, fillers, analyzed, wpm, clarity, confidence, pace) -> SessionRow:
    return SessionRow(words, fillers, analyzed, wpm, clarity, confidence, pace, PROMPT, MODEL)


# En orden cronológico de grabación, como las devuelve `fetch_sessions`.
REAL_SESSIONS = [
    _real(10, 8, True, 52.6, 10, 15, 20),  # 10 palabras: por debajo del mínimo
    _real(15, 0, False, 50.5, 85, 80, 20),  # en inglés
    _real(73, 17, True, 140.8, 65, 55, 90),  # sesión 01
    _real(54, 12, True, 147.5, 70, 55, 90),  # sesión 02
    _real(1, 0, False, 3.0, None, None, 5),  # sesión 03: silencio, Whisper dijo "you"
    _real(113, 27, True, 144.4, 68, 55, 88),  # sesión 04
    _real(47, 15, True, 138.2, 60, 45, 88),  # sesión 05
    _real(51, 8, True, 137.6, 70, 65, 88),  # sesión 06
    _real(58, 1, True, 155.3, 85, 78, 88),  # sesión 07
    _real(78, 0, True, 159.9, 90, 92, 85),  # sesión 08
    _real(55, 0, True, 136.5, 95, 90, 92),  # sesión 09
    _real(127, 0, True, 143.6, 78, 88, 95),  # sesión 10
]


def _build(rows, *, prompt_version=PROMPT, model=MODEL):
    return build_baseline(rows, prompt_version=prompt_version, analysis_model=model)


# ---------------------------------------------------------------------------
# Datos reales
# ---------------------------------------------------------------------------


class TestRealData:
    """Los resultados que se contrastaron con SQL a mano en el Bloque 7.1."""

    def test_counts_exclude_short_foreign_and_silence(self):
        result = _build(REAL_SESSIONS)
        assert result.status == BaselineStatus.READY
        assert result.valid_sessions == 9
        assert result.excluded_sessions == 3
        assert result.sessions_needed == 0

    def test_filler_rate(self):
        fr = _build(REAL_SESSIONS).metrics.filler_rate
        assert (fr.baseline, fr.recent, fr.delta) == (23.4, 0.0, -23.4)
        assert fr.trend == FillerTrend.IMPROVING
        assert (fr.baseline_samples, fr.recent_samples) == (5, 3)

    def test_words_per_minute(self):
        wpm = _build(REAL_SESSIONS).metrics.words_per_minute
        assert (wpm.baseline, wpm.recent, wpm.delta) == (141.7, 146.7, 5.0)
        assert wpm.trend == PaceTrend.STABLE

    def test_scores(self):
        scores = _build(REAL_SESSIONS).scores
        assert scores.status == BaselineStatus.READY
        assert scores.clarity.trend == ScoreTrend.MUCH_BETTER
        assert scores.confidence.trend == ScoreTrend.MUCH_BETTER
        assert scores.pace.trend == ScoreTrend.SIMILAR

    def test_silence_hallucination_does_not_enter_baseline(self):
        # Sin el mínimo de palabras, el `pace=5` del silencio entraría en la
        # ventana de scores. Con él, la quinta sesión de la línea base es la 06.
        assert not bs.is_valid_session(REAL_SESSIONS[4])
        valid = [r for r in REAL_SESSIONS if bs.is_valid_session(r)]
        baseline, _ = bs.split_windows(valid)
        assert baseline[-1] == REAL_SESSIONS[7]


# ---------------------------------------------------------------------------
# Estados
# ---------------------------------------------------------------------------


def test_user_without_sessions_is_collecting_not_an_error():
    result = _build([])
    assert result.status == BaselineStatus.COLLECTING
    assert (result.valid_sessions, result.excluded_sessions, result.sessions_needed) == (0, 0, 5)
    assert result.metrics is None
    assert result.scores.status == BaselineStatus.COLLECTING
    assert result.scores.clarity is None


def test_new_prompt_version_resets_scores_but_not_metrics():
    result = _build(REAL_SESSIONS, prompt_version="v2")
    assert result.status == BaselineStatus.READY
    assert result.metrics is not None
    assert result.scores.status == BaselineStatus.COLLECTING
    assert result.scores.valid_sessions == 0
    assert result.scores.clarity is None


def test_new_analysis_model_resets_scores_too():
    result = _build(REAL_SESSIONS, model="openai/gpt-oss-20b")
    assert result.status == BaselineStatus.READY
    assert result.scores.status == BaselineStatus.COLLECTING


def test_exactly_baseline_size_is_ready_without_recent():
    result = _build([row()] * 5)
    assert result.status == BaselineStatus.READY
    fr = result.metrics.filler_rate
    assert (fr.recent_samples, fr.recent, fr.trend) == (0, None, FillerTrend.INSUFFICIENT_DATA)


def test_one_recent_session_is_insufficient():
    result = _build([row()] * 6)
    assert result.metrics.words_per_minute.recent_samples == 1
    assert result.metrics.words_per_minute.trend == PaceTrend.INSUFFICIENT_DATA
    assert result.scores.clarity.trend == ScoreTrend.INSUFFICIENT_DATA


# ---------------------------------------------------------------------------
# Validez y ventanas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("session", "expected"),
    [
        (row(words=14), False),
        (row(words=15), True),
        (row(analyzed=False), False),
    ],
    ids=["14-palabras", "15-palabras", "no-espanol"],
)
def test_is_valid_session(session, expected):
    assert bs.is_valid_session(session) is expected


@pytest.mark.parametrize(
    ("total", "expected_baseline", "expected_recent"),
    [
        (12, [0, 1, 2, 3, 4], [9, 10, 11]),
        (7, [0, 1, 2, 3, 4], [5, 6]),
        (5, [0, 1, 2, 3, 4], []),
        (3, [0, 1, 2], []),
    ],
)
def test_split_windows_never_overlap(total, expected_baseline, expected_recent):
    baseline, recent = bs.split_windows(list(range(total)))
    assert baseline == expected_baseline
    assert recent == expected_recent


# ---------------------------------------------------------------------------
# Nulos y mínimos de muestra
# ---------------------------------------------------------------------------


def test_null_scores_are_ignored_per_metric():
    # Claridad solo en 2 de las 5 de la línea base: no llega al mínimo de 3.
    baseline = [row(clarity=None)] * 3 + [row(clarity=60)] * 2
    result = _build(baseline + [row(clarity=90)] * 3)
    assert result.scores.clarity.baseline_samples == 2
    assert result.scores.clarity.trend == ScoreTrend.INSUFFICIENT_DATA
    # Las demás métricas de esas mismas sesiones no se ven afectadas.
    assert result.scores.confidence.trend == ScoreTrend.SIMILAR


def test_null_wpm_does_not_affect_filler_rate():
    # Ritmo en 1 de las 3 recientes (Groq no dio la duración): por debajo del
    # mínimo de 2. Las muletillas de esas sesiones siguen contando.
    result = _build([row()] * 5 + [row(wpm=None)] * 2 + [row()])
    wpm = result.metrics.words_per_minute
    assert (wpm.recent_samples, wpm.recent, wpm.trend) == (1, None, PaceTrend.INSUFFICIENT_DATA)
    assert result.metrics.filler_rate.trend == FillerTrend.STABLE


def test_filler_rate_is_pooled_not_mean_of_rates():
    # Media de tasas: (13,3 % + 0 %) / 2 = 6,7. Conjunta: 2 / 135 = 1,48.
    rate, samples = bs.filler_rate([row(words=15, fillers=2), row(words=120, fillers=0)])
    assert rate == pytest.approx(100 * 2 / 135)
    assert samples == 2


# ---------------------------------------------------------------------------
# Umbrales de tendencia
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("baseline", "recent", "expected"),
    [
        (3.0, 2.1, FillerTrend.STABLE),  # Δ -0,9: manda el suelo de 1,0
        (3.0, 2.0, FillerTrend.IMPROVING),  # Δ -1,0: el umbral es estricto
        (20.0, 22.9, FillerTrend.STABLE),  # Δ +2,9: manda el 15 % (= 3,0)
        (20.0, 23.0, FillerTrend.WORSENING),  # Δ +3,0
        (None, 2.0, FillerTrend.INSUFFICIENT_DATA),
    ],
)
def test_filler_trend(baseline, recent, expected):
    assert bs.filler_trend(baseline, recent) == expected


@pytest.mark.parametrize(
    ("baseline", "recent", "expected"),
    [
        (140.0, 149.9, PaceTrend.STABLE),
        (140.0, 150.0, PaceTrend.FASTER),
        (140.0, 130.0, PaceTrend.SLOWER),
        (140.0, None, PaceTrend.INSUFFICIENT_DATA),
    ],
)
def test_pace_trend(baseline, recent, expected):
    assert bs.pace_trend(baseline, recent) == expected


@pytest.mark.parametrize(
    ("baseline", "recent", "expected"),
    [
        (60, 67.9, ScoreTrend.SIMILAR),
        (60, 68, ScoreTrend.BETTER),
        (60, 74.9, ScoreTrend.BETTER),
        (60, 75, ScoreTrend.MUCH_BETTER),
        (60, 45.1, ScoreTrend.WORSE),
        (60, 45, ScoreTrend.MUCH_WORSE),
        (None, 60, ScoreTrend.INSUFFICIENT_DATA),
    ],
)
def test_score_trend(baseline, recent, expected):
    assert bs.score_trend(baseline, recent) == expected


# ---------------------------------------------------------------------------
# Regresión: coma flotante en las fronteras
#
# Antes de redondear el delta, 48 pares de ritmo con `delta` mostrado de +10,0
# (p. ej. 54,1 -> 64,1, que en coma flotante da 9,999...) salían `stable`, y 6
# de muletillas en el umbral de 1,0. Las cifras que llegan a las tendencias
# tienen un decimal, así que se recorren todas las de un rango realista.
# ---------------------------------------------------------------------------


def _tenths(start: float, stop: float) -> list[float]:
    return [n / 10 for n in range(round(start * 10), round(stop * 10))]


def test_pace_exactly_10_apart_is_never_stable():
    fallos = [
        (b, round(b + d, 1))
        for b in _tenths(20.0, 300.0)
        for d in (10.0, -10.0)
        if bs.pace_trend(b, round(b + d, 1)) == PaceTrend.STABLE
    ]
    assert fallos == []


def test_pace_9_9_apart_is_always_stable():
    fallos = [
        b for b in _tenths(20.0, 300.0) if bs.pace_trend(b, round(b + 9.9, 1)) != PaceTrend.STABLE
    ]
    assert fallos == []


def test_filler_exactly_on_absolute_floor_is_never_stable():
    # Por debajo de 6,7 el 15 % es menor que 1,0 y manda el suelo absoluto.
    fallos = [
        b for b in _tenths(1.0, 6.7) if bs.filler_trend(b, round(b - 1.0, 1)) == FillerTrend.STABLE
    ]
    assert fallos == []


def test_filler_exactly_on_relative_threshold_is_never_stable():
    # Solo las líneas base cuyo 15 % cae justo en un decimal (20,0 -> 3,0...).
    exactas = [b for b in _tenths(6.7, 40.0) if round(0.15 * b, 6) == round(0.15 * b, 1)]
    assert exactas, "el rango debería contener fronteras exactas"
    fallos = [
        b
        for b in exactas
        if bs.filler_trend(b, round(b + round(0.15 * b, 1), 1)) == FillerTrend.STABLE
    ]
    assert fallos == []


# ---------------------------------------------------------------------------
# Regresión: errores de conectividad
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        ConnectionRefusedError(),  # puerto cerrado / servidor parado
        socket.gaierror(),  # host que no resuelve
        ConnectionDoesNotExistError("connection was closed in the middle of operation"),
    ],
    ids=["puerto-cerrado", "host-inexistente", "credenciales-o-base-inexistente"],
)
def test_driver_errors_seen_in_practice_count_as_connectivity(exc):
    # Estas tres se vieron contra PostgreSQL real, y ninguna la envuelve
    # SQLAlchemy. Si alguna deja de estar en la tupla, vuelve el 500.
    assert isinstance(exc, CONNECTIVITY_ERRORS)
