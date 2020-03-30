from typing import NamedTuple, Iterator, Iterable  # , List, Tuple
from enum import Enum


HEBREW_LETTERS = ''.join(chr(c) for c in range(0x05d0, 0x05ea + 1))

NIQQUD = tuple(chr(c) for c in range(0x05b0, 0x05bc + 1)) + ('\u05b7',)
# print(''.join(x for s in zip('א' * len(NIQQUD), NIQQUD) for x in s))

SHIN_YEMANIT = '\u05c1'
SHIN_SMALIT = '\u05c2'
NIQQUD_SIN = (SHIN_YEMANIT, SHIN_SMALIT)

DAGESH = '\u05bc'  # note that DAGESH and SHURUK are one and the same

VALID_LETTERS = ['', ' ', '!', '"', "'", '(', ')', ',', '-', '.', ':', ';', '?',
                 'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'ך', 'כ', 'ל', 'ם', 'מ', 'ן', 'נ', 'ס', 'ע', 'ף',
                 'פ', 'ץ', 'צ', 'ק', 'ר', 'ש', 'ת']
SPECIAL_TOKENS = ['^', '@', 'H', 'O', '5']


def normalize(c):
    if c in VALID_LETTERS: return c
    if c in ['\n', '\t']: return ' '
    if c in ['־', '‒', '–', '—', '―', '−']: return '-'
    if c == '[': return '('
    if c == ']': return ')'
    if c in ['´', '‘', '’']: return "'"
    if c in ['“', '”', '״']: return '"'
    if c.isdigit(): return '5'
    if c == '…': return ','
    if c in ['ײ', 'װ', 'ױ']: return 'H'
    return 'O'


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
    return letter in list(NIQQUD)


def iterate_dotted_text(text: str) -> Iterator[HebrewItem]:
    n = len(text)
    text += '  '
    i = 0
    while i < n:
        dagesh = ''
        niqqud = ''
        sin = ''
        letter = normalize(text[i])
        i += 1
        if is_hebrew_letter(letter):
            maybe_dagesh = False
            while i < n and (is_niqqud(text[i]) or text[i] == DAGESH or text[i] in NIQQUD_SIN):
                if text[i] == DAGESH:
                    if letter == 'ו':
                        maybe_dagesh = True
                    else:
                        dagesh = text[i]
                elif text[i] in NIQQUD_SIN:
                    sin = text[i]
                elif text[i] == '\u05b9' and niqqud:  # fix HOLAM where there should be a SIN:
                    sin = SHIN_SMALIT
                else:
                    if niqqud == '\u05b9':  # fix HOLAM where there should be a SIN:
                        sin = SHIN_SMALIT
                    niqqud = text[i]
                i += 1
            if maybe_dagesh:
                if niqqud:
                    dagesh = DAGESH
                else:
                    niqqud = DAGESH
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
