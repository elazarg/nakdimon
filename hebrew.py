from collections import defaultdict, Counter
from typing import NamedTuple, Iterator, Iterable, List, Tuple


HEBREW_LETTERS = [chr(c) for c in range(0x05d0, 0x05ea + 1)]

BEGED_KEFET = ['ב', 'ג', 'ד', 'כ', 'פ', 'ת']

NIQQUD = [chr(c) for c in range(0x05b0, 0x05bc + 1)] + ['\u05b7']

HOLAM = '\u05b9'

SHIN_YEMANIT = '\u05c1'
SHIN_SMALIT = '\u05c2'
NIQQUD_SIN = [SHIN_YEMANIT, SHIN_SMALIT]

DAGESH_LETTER = '\u05bc'
DAGESH = [DAGESH_LETTER]  # note that DAGESH and SHURUK are one and the same

ANY_NIQQUD = NIQQUD + NIQQUD_SIN + DAGESH

VALID_LETTERS = [' ', '!', '"', "'", '(', ')', ',', '-', '.', ':', ';', '?',
                 'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'ך', 'כ', 'ל', 'ם', 'מ', 'ן', 'נ', 'ס', 'ע', 'ף',
                 'פ', 'ץ', 'צ', 'ק', 'ר', 'ש', 'ת']
SPECIAL_TOKENS = ['H', 'O', '5']


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

    def __str__(self):
        return self.letter + self.dagesh + self.sin + self.niqqud

    def __repr__(self):
        return repr((self.letter, bool(self.dagesh), bool(self.sin), ord(self.niqqud or chr(0))))


class HebrewCharacter:
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
            while i < n and text[i] in ANY_NIQQUD:
                if text[i] == DAGESH_LETTER:
                    if letter == 'ו':
                        maybe_dagesh = True
                    else:
                        dagesh = text[i]
                elif text[i] in NIQQUD_SIN:
                    sin = text[i]
                elif text[i] == HOLAM and niqqud:
                    # fix HOLAM where there should be a SIN
                    sin = SHIN_SMALIT
                else:
                    if niqqud == HOLAM:
                        # fix HOLAM where there should be a SIN
                        sin = SHIN_SMALIT
                    niqqud = text[i]
                i += 1
            if maybe_dagesh:
                if niqqud:
                    dagesh = DAGESH_LETTER
                else:
                    niqqud = DAGESH_LETTER
            if letter == 'ש' and not sin and niqqud == HOLAM and i < n and not is_hebrew_letter(text[i]):
                niqqud = ''
                sin = SHIN_SMALIT
        yield HebrewItem(letter, dagesh, sin, niqqud)


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


class Token:
    def __init__(self, items: List[HebrewItem]):
        self.items = items

    def __str__(self):
        return ''.join(str(c) for c in self.items)

    def __repr__(self):
        return 'Token(' + repr(self.items) + ')'

    def strip_nonhebrew(self) -> 'Token':
        start = 0
        end = len(self.items) - 1
        while self.items[start].letter not in HEBREW_LETTERS:
            start += 1
            if start >= len(self.items):
                return Token([])
        while self.items[end].letter not in HEBREW_LETTERS:
            end -= 1
        return Token(self.items[start:end+1])

    def __bool__(self):
        return bool(self.items)

    def to_undotted(self):
        return ''.join(str(c.letter) for c in self.items)

    def is_undotted(self):
        return len(self.items) > 1 and all(c.niqqud == '' for c in self.items)


def tokenize_into(tokens_list: List[Token], char_iterator: Iterator[HebrewItem]) -> Iterator[HebrewItem]:
    current = []
    for c in char_iterator:
        if c.letter.isspace() or c.letter == '-':
            if current:
                tokens_list.append(Token(current))
            current = []
        else:
            current.append(c)
        yield c
    tokens_list.append(Token(current))


def iterate_file(path):
    with open(path, encoding='utf-8') as f:
        text = ' '.join(f.read().split())
        yield from iterate_dotted_text(text)


def tokenize(iterator: Iterator[HebrewItem]) -> List[Token]:
    tokens = []
    _ = list(tokenize_into(tokens, iterator))
    return tokens


def collect_wordmap(tokens: Iterable[Token]):
    word_dict = defaultdict(Counter)
    for token in tokens:
        word_dict[token.to_undotted()][str(token)] += 1
    return word_dict


if __name__ == '__main__':
    tokens = tokenize(iterate_file('texts/agadot.txt'))
    stripped_tokens = [token.strip_nonhebrew() for token in tokens if token.strip_nonhebrew()]
    # word_dict = collect_wordmap(stripped_tokens)
    # for k, v in sorted(word_dict.items(), key=lambda kv: (len(kv[1]), sum(kv[1].values()))):
    #     print(k, ':', str(v).replace('Counter', ''))
    # print(len(word_dict))
    for t in stripped_tokens:
        if any(item.letter == 'ש' and not item.sin for item in t.items):
            print(t)
