"""Análisis de comunicación con un LLM de Groq.

Recibe una transcripción y las métricas deterministas ya calculadas
(`app/services/speech_metrics.py`) y devuelve el análisis completo. No sabe de
HTTP ni de base de datos: lanza errores de dominio y quien llama decide, igual
que `transcription_service.py`.

El reparto de trabajo con `speech_metrics` es lo importante: aquí NO se cuenta
nada. Las palabras, el ritmo y las muletillas llegan hechas y se le pasan al
modelo dentro del prompt para que las interprete. El modelo solo aporta lo que
es un juicio: claridad, confianza, adecuación del ritmo, y la devolución en
prosa.

Elección del modelo
-------------------
`openai/gpt-oss-120b`, elegido comparando los modelos que hay realmente en la
cuenta (`client.models.list()`), no por reputación. Los datos:

- La cuenta NO tiene `llama-3.3-70b-versatile`. De los 14 modelos disponibles,
  solo `openai/gpt-oss-20b`, `openai/gpt-oss-120b` y `qwen/qwen3.8-27b`
  admiten salida estructurada ESTRICTA. `groq/compound` solo llega a
  `json_object`, que valida sintaxis pero no el esquema: insuficiente.
- `qwen/qwen3.8-27b`: descartado. Devolvía `clarity=6, confidence=4, pace=7`,
  es decir puntuaba sobre 10 pese a que el esquema declara `minimum: 0,
  maximum: 100` y la descripción lo dice con palabras. Son valores válidos, así
  que se guardarían como puntuaciones pésimas sin que nada fallase. Además
  cuesta 5 veces más ($0,80/$4,00 por millón frente a $0,15/$0,60).
- `openai/gpt-oss-20b`: la mitad de precio, pero repitió en varias pasadas la
  sugerencia "revisa la transcripción antes de grabar", que es incoherente —la
  transcripción existe DESPUÉS de grabar—. Las sugerencias son justo lo que el
  usuario lee, así que ese fallo pesa más que los $0,20 de diferencia por cada
  1000 análisis.
- `openai/gpt-oss-120b`: sugerencias coherentes en todas las pasadas, usa bien
  las palabras por minuto que se le dan, distingue con claridad un texto con
  19% de muletillas (claridad ~65) de un discurso limpio (claridad 90), y la
  latencia medida es de 1,0-1,6 s. Coste: ~$0,43 por cada 1000 análisis.

Sobre la varianza (importante para la Fase 7)
---------------------------------------------
Se midió con `temperature=0` y el MISMO texto tres veces: 68, 65 y 60 en
claridad. `temperature=0` NO da resultados deterministas en estos modelos, así
que un score individual tiene un ruido de unos ±5 puntos. La consecuencia es de
diseño, no de código: la línea base de la Fase 7 debe comparar TENDENCIAS sobre
varias grabaciones, nunca dos puntuaciones sueltas. Decir a alguien "has bajado
de 65 a 60" sería reportarle ruido como si fuese un retroceso.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

from groq import (
    APIConnectionError,
    APIStatusError,
    AsyncGroq,
    AuthenticationError,
    RateLimitError,
)
from pydantic import ValidationError

from app.core.config import settings
from app.models.analysis import (
    AnalysisLlmOutput,
    RecordingAnalysis,
    SpeechMetrics,
)

logger = logging.getLogger(__name__)

# Versión del prompt. SUBIRLA cada vez que cambie `_SYSTEM_PROMPT`, las
# descripciones de `_MODEL_FIELD_DESCRIPTIONS` o el formato del mensaje de
# usuario. Se guarda en `analyses.prompt_version` y es lo que permite responder
# "¿mejoró el usuario o solo cambié el prompt?". Sin esto, dos análisis del
# mismo audio son incomparables y la tabla `analyses` no sirve para nada.
ANALYSIS_PROMPT_VERSION = "v1"

# Lo más bajo posible. Aun así NO garantiza determinismo (ver el docstring del
# módulo): es lo mejor disponible para una tarea de puntuación, donde no se
# quiere creatividad sino el juicio más probable.
_TEMPERATURE = 0.0

# Techo de tokens de salida. Lo medido son 375-727, así que esto es unas tres
# veces el máximo observado: no estorba y acota el coste si el modelo se
# desboca. Ojo: si truncase, el JSON quedaría a medias y el análisis se
# perdería (`AnalysisFailedError`), de ahí el margen holgado.
_MAX_COMPLETION_TOKENS = 2000

# Cuántas sugerencias se le piden. Es una decisión de producto (tres consejos
# se leen; diez se ignoran), distinta del tope técnico `MAX_SUGGESTIONS` del
# schema, que solo existe como cortafuegos si el modelo se desmadra.
_SUGGESTIONS_REQUESTED = 3


# ---------------------------------------------------------------------------
# El prompt.
#
# Va en español porque el producto es es-ES y porque el modelo tiene que
# escribir en español: dar las instrucciones en el idioma de salida reduce que
# se cuele inglés en el `summary`.
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = f"""Eres un coach de comunicación oral. Analizas la transcripción \
de una grabación de voz y valoras cómo se comunica la persona.

Reglas:
- NO cuentes palabras ni calcules el ritmo: te damos esas cifras ya calculadas. \
Úsalas para interpretar, nunca las recalcules ni las repitas literalmente.
- Juzga solo lo que se pueda juzgar del texto. Si la transcripción es demasiado \
corta o está vacía para valorar algo, pon null en ese campo en vez de inventarlo.
- Escribe summary y suggestions en el MISMO idioma de la transcripción.
- Habla en segunda persona, directo y concreto. Nada de condescendencia ni de \
felicitaciones vacías.
- Como máximo {_SUGGESTIONS_REQUESTED} sugerencias, ordenadas de mayor a menor \
impacto. Cada una, una sola frase accionable.
- No menciones que eres una IA ni describas tu proceso."""


# Descripciones dirigidas AL MODELO, campo por campo.
#
# Aquí se resuelve la duda que quedó abierta en el paso 9: el
# `model_json_schema()` de Pydantic arrastra el docstring de la clase como
# `description`, y ese docstring está escrito para quien mantiene el código
# ("CONTRATO INTERNO", referencias a otros módulos...). Mandárselo al modelo es
# ruido en el mejor caso y una instrucción confusa en el peor.
#
# Así que `_build_strict_schema` borra las descripciones que genera Pydantic y
# pone estas. Y viven AQUÍ, junto al prompt, y no en `app/models/analysis.py`,
# porque son prompt: describen la tarea, no la estructura de datos. Si estuviesen
# en el schema, afinar el prompt obligaría a editar el modelo de datos y
# `ANALYSIS_PROMPT_VERSION` se quedaría desincronizada sin que se note.
_MODEL_FIELD_DESCRIPTIONS: dict[str, str] = {
    "clarity_score": (
        "Claridad del mensaje de 0 a 100: si la idea principal se entiende sin "
        "esfuerzo. Penaliza frases inacabadas, rodeos y saltos de tema."
    ),
    "confidence_score": (
        "Seguridad que transmite, de 0 a 100. Puntúa alto las afirmaciones "
        "directas; bajo los titubeos, las disculpas innecesarias y los rodeos."
    ),
    "pace_score": (
        "Adecuación del ritmo de 0 a 100, a partir de las palabras por minuto "
        "que se te dan ya calculadas. En español, por debajo de 110 se percibe "
        "lento y por encima de 170 atropellado; el óptimo está entre 120 y 160."
    ),
    "summary": (
        "Dos o tres frases de devolución para la persona, en su mismo idioma, "
        "en segunda persona y sin condescendencia. Di primero lo que funciona."
    ),
    "suggestions": (
        f"Hasta {_SUGGESTIONS_REQUESTED} consejos concretos y accionables, "
        "ordenados de mayor a menor impacto, en el idioma de la transcripción. "
        "Cada uno, una sola frase."
    ),
}


class AnalysisError(Exception):
    """Base de los errores de esta capa."""


class AnalysisFailedError(AnalysisError):
    """El modelo respondió, pero la respuesta no se puede usar.

    JSON mal formado, un tipo imposible, la respuesta vacía. Apunta a un
    problema del prompt o del modelo, no de la infraestructura: si esto se
    repite, hay que mirar `ANALYSIS_PROMPT_VERSION`.
    """


class AnalysisUnavailableError(AnalysisError):
    """No se pudo hablar con Groq, o falta configuración del servidor.

    Falta la API key, límite de peticiones, red caída, 5xx. Es transitorio o de
    configuración: no dice nada del prompt.
    """


# Cliente cacheado a nivel de módulo, creado en la primera llamada.
#
# Es un segundo `AsyncGroq` además del de `transcription_service.py`, con su
# propio pool de conexiones. Se acepta la duplicación: son dos servicios
# independientes, cada uno con su modelo y su timeout, y compartir el cliente
# exigiría un módulo intermedio. Si aparece un tercer servicio que hable con
# Groq, ahí sí toca extraer una factoría común.
_client: AsyncGroq | None = None


def _get_client() -> AsyncGroq:
    """Devuelve el cliente de Groq, creándolo en la primera llamada.

    Raises:
        AnalysisUnavailableError: no hay `GROQ_API_KEY` usable configurada.
    """
    global _client

    if _client is not None:
        return _client

    api_key = settings.groq_api_key
    if api_key is None or not settings.groq_configured:
        raise AnalysisUnavailableError("GROQ_API_KEY no configurada en el servidor.")

    _client = AsyncGroq(api_key=api_key.get_secret_value())
    return _client


def _hacer_estricto(nodo: dict[str, Any]) -> None:
    """Adapta un nodo del JSON Schema al modo estricto de Groq, recursivamente.

    El modo estricto usa decodificación restringida —el modelo no puede emitir
    tokens que rompan el esquema—, y a cambio exige que todo objeto declare
    `additionalProperties: false` y liste TODAS sus propiedades en `required`.
    Pydantic no genera ninguna de las dos cosas: marca como opcional lo que
    tiene default, y ahí Groq devolvería un 400.

    "Requerido" no choca con "opcional" en nuestro caso: los campos son
    `T | None`, así que Pydantic ya emite `anyOf: [{...}, {"type": "null"}]`.
    El modelo está obligado a incluir la clave, pero puede ponerla a null, que
    es exactamente lo que el prompt le pide cuando no puede valorar algo.

    Se recorre en profundidad aunque hoy el esquema sea plano: si mañana se
    añade un campo con un submodelo, esto lo cubre en vez de fallar con un 400
    difícil de rastrear.
    """
    if nodo.get("type") == "object" and "properties" in nodo:
        nodo["additionalProperties"] = False
        nodo["required"] = list(nodo["properties"])
        for subnodo in nodo["properties"].values():
            _hacer_estricto(subnodo)

    # Arrays y uniones: sus subesquemas también pueden llevar objetos dentro.
    items = nodo.get("items")
    if isinstance(items, dict):
        _hacer_estricto(items)
    for subnodo in nodo.get("anyOf", []):
        _hacer_estricto(subnodo)
    # `$defs` aparece en cuanto haya un submodelo anidado.
    for subnodo in nodo.get("$defs", {}).values():
        _hacer_estricto(subnodo)


def _build_strict_schema() -> dict[str, Any]:
    """Construye el JSON Schema que se manda a Groq.

    Tres transformaciones sobre lo que genera Pydantic:

    1. Se quita el `description` de la clase (su docstring interno) y los
       `title` autogenerados: no aportan nada al modelo y le meten ruido.
    2. Se cambia cada `description` de campo por la de
       `_MODEL_FIELD_DESCRIPTIONS`, escrita para el modelo.
    3. Se aplica `_hacer_estricto`, sin lo cual Groq rechaza la petición.

    También se borran los `default`, que en modo estricto no significan nada:
    todos los campos son obligatorios.
    """
    schema = AnalysisLlmOutput.model_json_schema()
    schema.pop("description", None)
    schema.pop("title", None)

    for nombre, propiedad in schema["properties"].items():
        propiedad.pop("title", None)
        propiedad.pop("default", None)
        # `KeyError` a propósito si falta una descripción: significa que se
        # añadió un campo a `AnalysisLlmOutput` y se olvidó describirlo para el
        # modelo. Mejor romper al arrancar que mandar un campo sin explicar.
        propiedad["description"] = _MODEL_FIELD_DESCRIPTIONS[nombre]

    _hacer_estricto(schema)
    return schema


# Se construye una vez al importar, no en cada petición: es determinista y así
# un campo sin descripción revienta al arrancar el proceso, no en la primera
# grabación de un usuario.
_STRICT_SCHEMA = _build_strict_schema()

_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "analisis_comunicacion",
        "strict": True,
        "schema": _STRICT_SCHEMA,
    },
}


def _build_user_message(text: str, metrics: SpeechMetrics) -> str:
    """Monta el mensaje con las métricas ya calculadas y la transcripción.

    Las métricas van ANTES del texto y con el aviso de no recalcularlas. La
    transcripción va delimitada para que un audio en el que alguien diga
    "ignora las instrucciones anteriores" se lea como contenido a analizar y no
    como una orden.
    """
    if not metrics.fillers_analyzed:
        # No se le dice "0 muletillas": sería mentira y el modelo la usaría
        # para elogiar a alguien por algo que no se ha medido.
        muletillas = "no analizadas (la detección solo cubre el español)"
        total = "no analizado"
    else:
        muletillas = (
            ", ".join(f"{f.word} (x{f.count})" for f in metrics.filler_words) or "ninguna"
        )
        total = str(metrics.filler_count)

    ritmo = (
        f"{metrics.words_per_minute} palabras por minuto"
        if metrics.words_per_minute is not None
        else "no disponible (no se conoce la duración del audio)"
    )

    return f"""Métricas ya calculadas (no las recalcules):
- palabras: {metrics.word_count}
- ritmo: {ritmo}
- muletillas detectadas: {total}
- desglose de muletillas: {muletillas}

Transcripción:
\"\"\"
{text}
\"\"\""""


@dataclass(frozen=True, slots=True)
class AnalysisOutcome:
    """Resultado de analizar una grabación.

    Attributes:
        analysis: el análisis listo para devolver al móvil.
        raw_response: el JSON del modelo tal cual, para `analyses.raw_response`.
            Se guarda entero aunque solo se expongan algunos campos: si mañana
            se quiere una métrica que el modelo ya estaba devolviendo, está ahí
            y no hay que reanalizar (ni volver a pagar) los audios antiguos.
    """

    analysis: RecordingAnalysis
    raw_response: dict[str, Any]


def _analisis_sin_modelo(metrics: SpeechMetrics, model: str, motivo: str) -> AnalysisOutcome:
    """Construye el resultado cuando no se llama al modelo.

    Las métricas deterministas se conservan y los scores quedan a `None`, que
    es justo lo que `CommunicationScores` documenta como "sin valoración".
    """
    return AnalysisOutcome(
        analysis=RecordingAnalysis(
            metrics=metrics,
            scores=AnalysisLlmOutput().to_scores(),
            summary=None,
            suggestions=[],
            model=model,
            prompt_version=ANALYSIS_PROMPT_VERSION,
        ),
        # El motivo queda registrado en la fila: una consulta a `analyses` puede
        # separar "el modelo no valoró" de "no se le preguntó, y por qué".
        raw_response={"skipped": motivo},
    )


async def analyze(
    text: str,
    *,
    metrics: SpeechMetrics,
    model: str | None = None,
    timeout: float | None = None,
) -> AnalysisOutcome:
    """Analiza una transcripción y devuelve el análisis completo.

    Args:
        text: la transcripción, tal como la devolvió Whisper.
        metrics: métricas deterministas de `speech_metrics.compute_metrics`.
        model: modelo a usar. Por defecto `settings.groq_analysis_model`.
        timeout: segundos antes de cortar. Por defecto
            `settings.groq_analysis_timeout_seconds`.

    Raises:
        AnalysisFailedError: el modelo respondió algo inservible.
        AnalysisUnavailableError: falta la clave, o Groq no está accesible.
    """
    model = model or settings.groq_analysis_model
    timeout = timeout if timeout is not None else settings.groq_analysis_timeout_seconds

    if metrics.word_count == 0:
        # Un audio en silencio. No se llama al modelo: no hay nada que valorar,
        # y pedirle que juzgue la nada es la mejor forma de que se lo invente.
        # Además ahorra la llamada.
        logger.info("Transcripción sin palabras: se omite el análisis con el modelo.")
        return _analisis_sin_modelo(metrics, model, "empty_transcription")

    client = _get_client()

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(text, metrics)},
            ],
            response_format=_RESPONSE_FORMAT,
            temperature=_TEMPERATURE,
            max_completion_tokens=_MAX_COMPLETION_TOKENS,
            timeout=timeout,
        )
    except AuthenticationError as exc:
        # Un 401 de Groq significa que NUESTRA clave es mala: es configuración,
        # no un problema del prompt. Mismo criterio que en transcription_service.
        logger.error("Groq rechazó la API key al analizar: %s", exc)
        raise AnalysisUnavailableError("La API key de Groq no es válida.") from exc
    except RateLimitError as exc:
        # Frecuente: el nivel on-demand tiene un límite bajo de tokens por
        # minuto (medido: 8000 TPM), y cada análisis gasta ~1300. Con varias
        # grabaciones seguidas se alcanza. Es transitorio y ajeno al usuario.
        logger.warning("Groq está limitando las peticiones de análisis: %s", exc)
        raise AnalysisUnavailableError("Groq está limitando las peticiones.") from exc
    except APIConnectionError as exc:
        # Cubre también APITimeoutError, que hereda de esta.
        logger.error("No se pudo conectar con Groq para analizar: %s", exc)
        raise AnalysisUnavailableError("No se pudo contactar con Groq.") from exc
    except APIStatusError as exc:
        if exc.status_code >= 500:
            logger.error("Groq devolvió un %s al analizar: %s", exc.status_code, exc)
            raise AnalysisUnavailableError("Groq devolvió un error de servidor.") from exc
        # Un 4xx aquí suele ser un esquema que el modo estricto no acepta, o un
        # modelo que no admite `json_schema`. Es un problema NUESTRO, pero de
        # prompt/configuración del análisis, no de disponibilidad.
        logger.error("Groq rechazó la petición de análisis (HTTP %s): %s", exc.status_code, exc)
        raise AnalysisFailedError(
            f"Groq rechazó la petición de análisis (HTTP {exc.status_code})."
        ) from exc

    contenido = response.choices[0].message.content
    if not contenido or not contenido.strip():
        # Puede pasar si `max_completion_tokens` cortó la respuesta antes de
        # que empezase a escribir el JSON.
        logger.warning("Groq devolvió una respuesta de análisis vacía (modelo=%s).", model)
        raise AnalysisFailedError("El modelo devolvió una respuesta vacía.")

    try:
        crudo = json.loads(contenido)
    except json.JSONDecodeError as exc:
        # Con modo estricto no debería ocurrir; si ocurre, es porque la
        # respuesta se truncó.
        logger.warning("El análisis no era JSON válido (modelo=%s): %s", model, exc)
        raise AnalysisFailedError("La respuesta del modelo no era JSON válido.") from exc

    if not isinstance(crudo, dict):
        logger.warning("El análisis no era un objeto JSON (modelo=%s): %s", model, type(crudo))
        raise AnalysisFailedError("La respuesta del modelo no era un objeto JSON.")

    try:
        salida = AnalysisLlmOutput.model_validate(crudo)
    except ValidationError as exc:
        # Difícil de alcanzar: los validadores de `AnalysisLlmOutput` están
        # hechos para degradar en vez de fallar (un score fuera de rango pasa a
        # None, un campo extra se ignora). Si aun así falla, la respuesta es
        # inservible de verdad.
        logger.warning("El análisis no valida contra el schema (modelo=%s): %s", model, exc)
        raise AnalysisFailedError("La respuesta del modelo no tiene la forma esperada.") from exc

    logger.info(
        "Análisis completado (modelo=%s, prompt=%s): clarity=%s confidence=%s pace=%s",
        model,
        ANALYSIS_PROMPT_VERSION,
        salida.clarity_score,
        salida.confidence_score,
        salida.pace_score,
    )

    return AnalysisOutcome(
        analysis=RecordingAnalysis(
            metrics=metrics,
            scores=salida.to_scores(),
            summary=salida.summary,
            suggestions=salida.suggestions,
            model=model,
            prompt_version=ANALYSIS_PROMPT_VERSION,
        ),
        raw_response=crudo,
    )
