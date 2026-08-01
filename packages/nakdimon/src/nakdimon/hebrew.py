"""Core Hebrew text layer: normalization, cleanup, and dotted-text decomposition /
reconstruction. Driven entirely by spec/hebrew.json (packaged as data/hebrew.json) —
see that file for the semantics this implements.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import NamedTuple

from nakdimon.spec import load_spec

# "category applies but deliberately absent" marker (U+05BF).
RAFE = "ֿ"

_HEBREW_LO, _HEBREW_HI = 0x05D0, 0x05EA
_DAGESH_MARK = "ּ"
_SHIN_YEMANIT = "ׁ"
_SHIN_SMALIT = "ׂ"
_NIQQUD_LO, _NIQQUD_HI = 0x05B0, 0x05BB  # shva .. kubuts


def _is_hebrew_letter(c: str) -> bool:
    return _HEBREW_LO <= ord(c) <= _HEBREW_HI


class HebrewItem(NamedTuple):
    letter: str
    normalized: str
    dagesh: str
    sin: str
    niqqud: str


@lru_cache(maxsize=1)
def _normalization() -> tuple[frozenset[str], dict[str, str]]:
    norm = load_spec()["normalization"]
    return frozenset(norm["pass_through"]), norm["map"]


def normalize_char(c: str) -> str:
    pass_through, mapping = _normalization()
    if c in pass_through:
        return c
    if c in mapping:
        return mapping[c]
    if unicodedata.category(c) == "Nd":
        return "5"
    return "O"


def normalize_text(s: str) -> str:
    return "".join(normalize_char(c) for c in s)


@lru_cache(maxsize=1)
def _strip_chars() -> frozenset[str]:
    return frozenset(load_spec()["text_cleanup"]["strip"])


def clean_text(s: str) -> str:
    strip = _strip_chars()
    return "".join(c for c in s if c not in strip)


@lru_cache(maxsize=1)
def _remove_niqqud_re() -> re.Pattern[str]:
    # remove_niqqud.chars is the BODY of a regex character class (a range plus a
    # handful of individual marks) — spliced in as-is, no escaping needed or wanted.
    return re.compile(f"[{load_spec()['remove_niqqud']['chars']}]")


def remove_niqqud(s: str) -> str:
    return _remove_niqqud_re().sub("", s)


@lru_cache(maxsize=1)
def _decomposition_sets() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    decomp = load_spec()["decomposition"]
    return (
        frozenset(decomp["can_dagesh"]),
        frozenset(decomp["can_sin"]),
        frozenset(decomp["can_niqqud"]),
    )


def can_dagesh(letter: str) -> bool:
    return letter in _decomposition_sets()[0]


def can_sin(letter: str) -> bool:
    return letter in _decomposition_sets()[1]


def can_niqqud(letter: str) -> bool:
    return letter in _decomposition_sets()[2]


def decompose(dotted: str) -> list[HebrewItem]:
    """Order-tolerant parser for dotted text. Combining marks immediately following a
    Hebrew base letter are consumed in ANY order, last-wins per channel; items_to_text
    re-emits in NFC canonical order (niqqud, dagesh, sin-dot). Legacy v1 corpora use
    dagesh-sin-niqqud order, which this parser reads equally well (see spec["mark_order"]).
    """
    items: list[HebrewItem] = []
    n = len(dotted)
    i = 0
    while i < n:
        letter = dotted[i]
        i += 1

        dagesh = RAFE if can_dagesh(letter) else ""
        sin = RAFE if can_sin(letter) else ""
        niqqud = RAFE if can_niqqud(letter) else ""
        normalized = normalize_char(letter)

        if _is_hebrew_letter(letter):
            while i < n:
                mark = dotted[i]
                if mark == _DAGESH_MARK:
                    dagesh = mark
                elif mark in (_SHIN_YEMANIT, _SHIN_SMALIT):
                    sin = mark
                elif mark == RAFE or _NIQQUD_LO <= ord(mark) <= _NIQQUD_HI:
                    niqqud = mark
                else:
                    break
                i += 1
            # A dagesh on vav with no explicit niqqud is really the shuruk vowel.
            if letter == "ו" and dagesh == _DAGESH_MARK and niqqud == RAFE:
                dagesh = RAFE
                niqqud = _DAGESH_MARK

        items.append(HebrewItem(letter, normalized, dagesh, sin, niqqud))
    return items


def items_to_text(items: list[HebrewItem]) -> str:
    # NFC canonical emission (spec["mark_order"]): niqqud (ccc 10-20), dagesh (21),
    # sin-dot (24/25) — ascending combining class, so the result is an NFC fixpoint.
    joined = "".join(item.letter + item.niqqud + item.dagesh + item.sin for item in items)
    return joined.replace(RAFE, "")
