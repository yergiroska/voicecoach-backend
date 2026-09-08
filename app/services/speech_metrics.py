"""Métricas deterministas de una transcripción: palabras, ritmo y muletillas.

Todo lo que se calcula aquí es aritmética y coincidencia de patrones. Nada de
esto se le pide al modelo de lenguaje, y es una decisión, no una comodidad:
contar palabras o dividir para sacar palabras por minuto son operaciones que un
LLM hace peor, cobra por hacer y no repite igual dos veces. El modelo se reserva
para lo que sí es un juicio —claridad, confianza, ritmo adecuado— y estas cifras
se le pasan ya calculadas para que las interprete.

La consecuencia práctica es que estas métricas están disponibles SIEMPRE, aunque
la llamada al modelo falle. Es lo que permite que un análisis degradado siga
teniendo valor (ver `RecordingAnalysis` en `app/models/analysis.py`).

La detección de muletillas es SOLO de español (`SPANISH_FILLERS`). Para otros
idiomas no se ejecuta y `SpeechMetrics.fillers_analyzed` queda en `False`, para
que un cero no se confunda con "no había ninguna". El recuento de palabras y el
ritmo sí valen para cualquier idioma. Añadir una lista en inglés es una mejora
pendiente; hoy el producto es es-ES.

Este módulo no lanza excepciones: cualquier texto es analizable, incluida la
cadena vacía de un audio en silencio.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass

from app.models.analysis import FillerWord, SpeechMetrics

logger = logging.getLogger(__name__)

# Qué cuenta como palabra. `\w` con Unicode ya cubre acentos, ñ y ü, que es lo
# que importa en español. Se admiten guiones y apóstrofos INTERNOS para que
# "diez-doce" o "d'acord" no se partan en dos, pero no en los extremos (un
# guion de raya suelto no es una palabra).
#
# Los números cuentan como palabra: si alguien dice "dos mil veintiséis",
# Whisper puede transcribirlo como "2026", y no contarlo falsearía el ritmo.
_WORD_RE = re.compile(r"\w+(?:['’\-]\w+)*", re.UNICODE)

# Signos que marcan el final de una unidad de habla. Son la clave de la
# heurística de muletillas ambiguas: Whisper puntúa las muletillas con comas de
# forma bastante fiable ("Bueno, eh, hoy quiero...").
_DELIMITERS = frozenset(".,;:!?¡¿…()[]{}«»\"'“”‘’—–\n\r\t")

# Segundos por minuto. Nombrado para que la fórmula del ritmo se lea sola.
_SECONDS_PER_MINUTE = 60.0


@dataclass(frozen=True, slots=True)
class FillerPattern:
    """Una muletilla a buscar.

    Attributes:
        phrase: la forma escrita, en minúsculas y con espacios simples. Puede
            tener varias palabras ("o sea"); los espacios se tratan como
            "uno o más espacios" al buscar.
        ambiguous: si la expresión tiene además un uso legítimo frecuente. Ver
            `_es_muletilla` para lo que implica.
    """

    phrase: str
    ambiguous: bool


# ---------------------------------------------------------------------------
# La lista de muletillas.
#
# PARA AMPLIARLA: añadir una línea a la tupla. No hace falta tocar nada más —
# el patrón de búsqueda se construye a partir de esto, y las expresiones más
# largas tienen prioridad automáticamente, así que añadir "o sea que" no rompe
# el conteo de "o sea".
#
# El campo `ambiguous` es la decisión importante de cada entrada:
#
#   ambiguous=False -> la expresión no significa nada más. Se cuenta siempre
#                      que aparezca.
#   ambiguous=True  -> la expresión tiene un uso legítimo habitual, así que
#                      solo se cuenta cuando aparece AISLADA entre signos de
#                      puntuación. "Bueno, hoy quiero..." cuenta; "un vino
#                      bueno para la cena" no.
#
# Sin esa distinción el conteo sería inservible: `este`, `bueno`, `pues` o
# `nada` están constantemente en cualquier frase en español con su significado
# normal, y contarlos a ciegas inflaría el resultado hasta convertirlo en ruido.
# El precio de la heurística es perder alguna muletilla que Whisper no puntuó;
# se prefiere quedarse corto y ser creíble a exagerar.
# ---------------------------------------------------------------------------
SPANISH_FILLERS: tuple[FillerPattern, ...] = (
    # --- Sonidos de duda. No son palabras: si aparecen, son muletillas.
    FillerPattern("eh", ambiguous=False),
    FillerPattern("ehm", ambiguous=False),
    FillerPattern("em", ambiguous=False),
    FillerPattern("mm", ambiguous=False),
    FillerPattern("mmm", ambiguous=False),
    # --- Conectores vacíos. Casi nunca tienen otro uso.
    FillerPattern("o sea", ambiguous=False),
    FillerPattern("por así decirlo", ambiguous=False),
    FillerPattern("como que", ambiguous=False),
    FillerPattern("ya sabes", ambiguous=False),
    # --- Expresiones con doble uso: solo cuentan aisladas.
    # "este/esto" como relleno frente a los demostrativos ("este libro").
    FillerPattern("este", ambiguous=True),
    FillerPattern("esto", ambiguous=True),
    # "bueno" adjetivo, "pues" conjunción, "entonces" temporal...
    FillerPattern("bueno", ambiguous=True),
    FillerPattern("pues", ambiguous=True),
    FillerPattern("entonces", ambiguous=True),
    FillerPattern("digamos", ambiguous=True),
    FillerPattern("vale", ambiguous=True),
    FillerPattern("nada", ambiguous=True),
    FillerPattern("tipo", ambiguous=True),
    FillerPattern("en plan", ambiguous=True),
    FillerPattern("es que", ambiguous=True),
    FillerPattern("la verdad", ambiguous=True),
    FillerPattern("sabes", ambiguous=True),
    FillerPattern("digo", ambiguous=True),
)


# Cómo puede llegar "español" en `Transcription.language`. Whisper lo devuelve
# en inglés y con la inicial en mayúscula ("Spanish", verificado en la prueba
# end-to-end), pero se aceptan también los códigos ISO y los endónimos para no
# depender de un detalle del proveedor.
SPANISH_LANGUAGE_TAGS = frozenset(
    {"es", "spa", "spanish", "español", "espanol", "castellano", "castilian"}
)

# Qué hacer cuando Groq no informa del idioma. Se asume español porque es el
# único idioma que soporta el producto hoy, y porque `language` es opcional solo
# por prudencia: en la práctica Whisper siempre lo devuelve. Tratar el
# desconocido como "no español" desactivaría la función principal de la app por
# una rareza del proveedor.
#
# La contrapartida, y hay que tenerla presente: si algún día llega un `None` en
# un audio que no era español, se contarán muletillas españolas sobre ese texto
# y `fillers_analyzed` dirá `True`. Cambiar esta constante a False invierte el
# criterio y no hace falta tocar nada más.
ASSUME_SPANISH_WHEN_UNKNOWN = True


def language_is_unknown(language: str | None) -> bool:
    """Si Whisper no informó del idioma.

    Vive aparte de `is_spanish` porque la condición se usa dos veces: para
    decidir (allí) y para avisar en los logs (en `compute_metrics`). Duplicar
    un `if` así es la forma habitual de que las dos copias dejen de coincidir.
    """
    return language is None or not language.strip()


def is_spanish(language: str | None) -> bool:
    """Si el idioma informado por Whisper es español.

    Acepta "Spanish", "es", "es-ES", "es_419", "español"... Un `None` o una
    cadena vacía se resuelven según `ASSUME_SPANISH_WHEN_UNKNOWN`.
    """
    if language_is_unknown(language):
        return ASSUME_SPANISH_WHEN_UNKNOWN

    etiqueta = language.strip().casefold()
    if etiqueta in SPANISH_LANGUAGE_TAGS:
        return True

    # "es-ES" / "es_419" -> "es". Se compara la parte principal, no el sufijo
    # regional: el español de México y el de España comparten las muletillas
    # que hay en la lista.
    principal = re.split(r"[-_]", etiqueta, maxsplit=1)[0]
    return principal in SPANISH_LANGUAGE_TAGS


def _construir_patron(fillers: tuple[FillerPattern, ...]) -> re.Pattern[str]:
    """Compila un único patrón con todas las muletillas.

    Un solo patrón y no uno por muletilla: así el texto se recorre una vez y,
    sobre todo, las coincidencias no se solapan. Las alternativas se ordenan de
    más larga a más corta porque la alternancia de `re` se queda con la primera
    que encaja: sin ese orden, añadir "o sea que" a la lista haría que se
    contase como "o sea" y la palabra "que" quedase fuera de la coincidencia.

    `\\s+` entre palabras tolera espacios dobles o un salto de línea en medio
    de una expresión de varias palabras.
    """
    alternativas = sorted(
        (f.phrase for f in fillers),
        key=lambda frase: (-len(frase.split()), -len(frase)),
    )
    partes = [r"\s+".join(re.escape(palabra) for palabra in frase.split()) for frase in alternativas]

    # `(?<!\w)` y `(?!\w)` en vez de `\b`: hacen lo mismo aquí pero no dependen
    # de que la expresión empiece o acabe en carácter de palabra. Evitan que
    # "eh" se cuente dentro de "cheque".
    return re.compile(rf"(?<!\w)(?:{'|'.join(partes)})(?!\w)", re.IGNORECASE | re.UNICODE)


_FILLER_RE = _construir_patron(SPANISH_FILLERS)

# Mapa de forma normalizada -> definición, para saber qué entrada de la lista
# produjo cada coincidencia (`re` no lo dice cuando se usa alternancia).
_FILLER_BY_PHRASE: dict[str, FillerPattern] = {f.phrase: f for f in SPANISH_FILLERS}


def _normalizar(fragmento: str) -> str:
    """Pasa una coincidencia a la forma canónica de la lista.

    Minúsculas y espacios colapsados: "O  Sea" y "o\\nsea" son ambas "o sea".
    No se quitan los acentos, porque en español distinguen palabras.
    """
    return re.sub(r"\s+", " ", fragmento.strip().casefold())


def _aislada(text: str, inicio: int, fin: int) -> bool:
    """Si la coincidencia está delimitada por puntuación (o por los extremos).

    Se mira el primer carácter no blanco a cada lado. El inicio y el final del
    texto cuentan como delimitador: "Bueno, hoy..." tiene el arranque de la
    frase a la izquierda y una coma a la derecha, y es una muletilla de manual.
    """
    izquierda = inicio - 1
    while izquierda >= 0 and text[izquierda].isspace():
        izquierda -= 1
    if izquierda >= 0 and text[izquierda] not in _DELIMITERS:
        return False

    derecha = fin
    longitud = len(text)
    while derecha < longitud and text[derecha].isspace():
        derecha += 1
    if derecha < longitud and text[derecha] not in _DELIMITERS:
        return False

    return True


def _es_muletilla(text: str, match: re.Match[str], filler: FillerPattern) -> bool:
    """Decide si esta coincidencia concreta cuenta como muletilla."""
    if not filler.ambiguous:
        return True
    return _aislada(text, match.start(), match.end())


def count_words(text: str) -> int:
    """Cuenta las palabras de la transcripción.

    Cuenta TODAS las palabras, muletillas incluidas: es el total de lo que se
    dijo, y es también la base del ritmo. Una expresión como "o sea" suma dos
    palabras aquí y una sola muletilla en `find_fillers`.
    """
    return len(_WORD_RE.findall(text))


def find_fillers(text: str) -> list[FillerWord]:
    """Detecta las muletillas del texto, de la más frecuente a la menos.

    El empate se rompe por orden alfabético. No es un detalle estético: sin un
    criterio fijo, dos análisis del mismo texto podrían guardar la lista en
    distinto orden y parecer diferentes.
    """
    contador: Counter[str] = Counter()

    for match in _FILLER_RE.finditer(text):
        frase = _normalizar(match.group(0))
        filler = _FILLER_BY_PHRASE.get(frase)
        if filler is None:
            # No debería pasar: el patrón se construye desde la misma lista.
            # Si pasara (una entrada con mayúsculas o espacios raros), se
            # ignora en silencio antes que contar algo que no está en la lista.
            continue
        if _es_muletilla(text, match, filler):
            contador[filler.phrase] += 1

    return [
        FillerWord(word=frase, count=veces)
        for frase, veces in sorted(contador.items(), key=lambda par: (-par[1], par[0]))
    ]


def words_per_minute(word_count: int, duration_seconds: float | None) -> float | None:
    """Ritmo del habla, o `None` si no se puede calcular.

    Devuelve `None` cuando falta la duración (Groq no la garantiza) o cuando no
    es positiva. No se estima por otros medios: un ritmo inventado alimentaría
    la línea base de la Fase 7 con un dato falso, y `None` ya está previsto en
    `SpeechMetrics.words_per_minute`.

    Con duración válida y cero palabras devuelve `0.0`, que es el ritmo real de
    un audio en silencio, no una ausencia de dato.

    Se redondea a un decimal: la precisión de `duration` de Whisper no da para
    más, y un 120.09873920000001 en la respuesta solo aporta ruido.
    """
    if duration_seconds is None or duration_seconds <= 0:
        return None
    return round(word_count * _SECONDS_PER_MINUTE / duration_seconds, 1)


def compute_metrics(
    text: str,
    *,
    duration_seconds: float | None,
    language: str | None = None,
    recording_id: str | None = None,
) -> SpeechMetrics:
    """Calcula todas las métricas deterministas de una transcripción.

    Es el punto de entrada del módulo; el resto de funciones son públicas para
    poder usarlas por separado.

    Si `language` no es español, la detección de muletillas NO se ejecuta y el
    resultado lleva `fillers_analyzed=False`. No se devuelve simplemente una
    lista vacía: eso sería indistinguible de "se analizó y no había ninguna", y
    la interfaz mostraría un "0 muletillas" falso en un audio en otro idioma.
    El recuento de palabras y el ritmo se calculan igual, que no dependen del
    idioma.

    Devuelve directamente `SpeechMetrics` —el schema de `app/models/`— en lugar
    de un dataclass propio como hacen los otros servicios. Aquí duplicar la
    estructura para copiar cinco campos uno a uno no aportaría desacoplamiento,
    solo dos definiciones que se van desincronizando: estas métricas SON el
    contenido de ese schema.

    Args:
        text: la transcripción. Puede venir vacía (audio en silencio).
        duration_seconds: duración del audio según Whisper, o `None`.
        language: idioma detectado por Whisper (p. ej. "Spanish"), o `None`.
        recording_id: solo para poder identificar la grabación en el WARNING de
            idioma desconocido. No influye en ningún cálculo.
    """
    palabras = count_words(text)
    analizar_muletillas = is_spanish(language)

    if language_is_unknown(language):
        # El caso no debería darse —Whisper siempre ha informado del idioma en
        # las pruebas— y por eso se avisa en vez de tragárselo: si empieza a
        # aparecer con frecuencia, la suposición de `ASSUME_SPANISH_WHEN_UNKNOWN`
        # deja de ser razonable y hay que replantearla con datos.
        logger.warning(
            "Whisper no informó del idioma (recording_id=%s): se asume %s "
            "(ASSUME_SPANISH_WHEN_UNKNOWN=%s), así que la detección de "
            "muletillas %s. Si este aviso se repite, revisar la suposición en "
            "app/services/speech_metrics.py.",
            recording_id or "desconocido",
            "español" if ASSUME_SPANISH_WHEN_UNKNOWN else "no español",
            ASSUME_SPANISH_WHEN_UNKNOWN,
            "SÍ se ejecuta" if analizar_muletillas else "NO se ejecuta",
        )

    muletillas = find_fillers(text) if analizar_muletillas else []

    return SpeechMetrics(
        word_count=palabras,
        words_per_minute=words_per_minute(palabras, duration_seconds),
        filler_count=sum(m.count for m in muletillas),
        filler_words=muletillas,
        fillers_analyzed=analizar_muletillas,
    )
