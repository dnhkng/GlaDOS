import re
from typing import Any, Dict

from phonemizer.phonemize import phonemize
from unidecode import unidecode

from utils.numbers import normalize_numbers
from utils.symbols import phonemes_set

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_units(text):
    text = text.replace("°C", "degrees selsius")
    text = text.replace("°F", "degrees fahrenheit")
    text = text.replace("°c", "degrees selsius")
    text = text.replace("°f", "degrees fahrenheit")
    text = text.replace("°", "degrees")
    text = text.replace("hPa", "hecto pascals")
    text = text.replace("g/m³", "grams per cubic meter")
    text = text.replace("% (RH)", "percent relative humidity")
    return text


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def no_cleaners(text):
    return text


def english_cleaners(text):
    text = expand_units(text)
    text = unidecode(text)
    text = normalize_numbers(text)
    text = expand_abbreviations(text)
    return text


def to_phonemes(text: str, lang: str) -> str:
    phonemes = phonemize(
        text,
        language=lang,
        backend="espeak",
        strip=True,
        preserve_punctuation=True,
        with_stress=False,
        njobs=1,
        punctuation_marks=';:,.!?¡¿—…"«»“”()',
        language_switch="remove-flags",
    )
    phonemes = "".join([p for p in phonemes if p in phonemes_set])
    return phonemes


class Cleaner:
    def __init__(self, cleaner_name: str, use_phonemes: bool, lang: str) -> None:
        if cleaner_name == "english_cleaners":
            self.clean_func = english_cleaners
        elif cleaner_name == "no_cleaners":
            self.clean_func = no_cleaners
        else:
            raise ValueError(
                f"Cleaner not supported: {cleaner_name}! "
                f"Currently supported: ['english_cleaners', 'no_cleaners']"
            )
        self.use_phonemes = use_phonemes
        self.lang = lang

    def __call__(self, text: str) -> str:
        text = self.clean_func(text)
        if self.use_phonemes:
            text = to_phonemes(text, self.lang)
        text = collapse_whitespace(text)
        text = text.strip()
        return text

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Cleaner":
        return Cleaner(
            cleaner_name=config["preprocessing"]["cleaner_name"],
            use_phonemes=config["preprocessing"]["use_phonemes"],
            lang=config["preprocessing"]["language"],
        )
