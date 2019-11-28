from typing import NamedTuple, Iterator, Iterable  # , List, Tuple
from enum import Enum


HEBREW_LETTERS = ''.join(chr(c) for c in range(0x05d0, 0x05ea + 1))

NIQQUD = tuple(chr(c) for c in range(0x0591, 0x05c7 + 1))

NIQQUD_SIN = ('\u05c1', '\u05c2')

DAGESH = '\u05bc'  # note that DAGESH and SHURUK are one and the same


class HebrewItem(NamedTuple):
    letter: str
    dagesh: str
    sin: str
    niqqud: str


class HebrewCharacter(Enum):
    ALEF = 'א'
    BET = 'ב'
    GIMEL = 'ג'
    DALET = 'ד'
    HE = 'ה'
    VAV = 'ו'
    ZAYIN = 'ז'
    HET = 'ח'
    TET = 'ט'
    YOD = 'י'
    KAF = 'כ'
    KAF_SOFIT = 'ך'
    LAMED = 'ל'
    MEM = 'מ'
    MEM_SOFIT = 'ם'
    NUN = 'נ'
    NUN_SOFIT = 'ן'
    SAMECH = 'ס'
    AYIN = 'ע'
    PE = 'פ'
    PE_SOFIT = 'ף'
    TZADI = 'צ'
    TZADI_SOFIT = 'ץ'
    KOF = 'ק'
    RESH = 'ר'
    SHIN = 'ש'
    TAV = 'ת'


def is_hebrew_letter(letter: str) -> bool:
    return '\u05d0' <= letter <= '\u05ea'


def is_niqqud(letter: str) -> bool:
    return '\u0591' <= letter <= '\u05c7'


def iterate_dotted_text(text: str) -> Iterator[HebrewItem]:
    n = len(text)
    text += '  '
    i = 0
    while i < n:
        dagesh = ''
        niqqud = ''
        sin = ''
        letter = text[i]
        i += 1
        if is_hebrew_letter(letter):
            while i < n and is_niqqud(text[i]) or text[i] in NIQQUD_SIN:
                if text[i] == DAGESH and (letter != HebrewCharacter.VAV or is_niqqud(text[i + 1])):
                    dagesh = text[i]
                elif text[i] in NIQQUD_SIN:
                    sin = text[i]
                else:
                    niqqud = text[i]
                i += 1
        yield HebrewItem(letter, dagesh, sin, niqqud)


def hebrew_items_to_str(items: Iterable[HebrewItem]) -> str:
    return ''.join(x for tup in items for x in tup)


def split_by_length(heb_items, maxlen):
    assert maxlen > 1

    maxlen -= 1

    start = 0
    while start < len(heb_items):
        if len(heb_items) <= start + maxlen:
            yield heb_items[start:]
            break
        ub = maxlen
        while ub > 0 and heb_items[start + ub][0] != ' ':
            ub -= 1
        if ub == 0:
            ub = maxlen
        yield heb_items[start:start + ub]
        start += ub + 1
