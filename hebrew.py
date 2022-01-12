import itertools
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import NamedTuple, Iterator, Iterable, List, Tuple
from functools import lru_cache
import re

import utils


# "rafe" denotes a letter to which it would have been valid to add a diacritic of some category
# but instead it is decided not to. This makes the metrics less biased.
RAFE = '\u05BF'


class Niqqud:
    SHVA = '\u05B0'
    REDUCED_SEGOL = '\u05B1'
    REDUCED_PATAKH = '\u05B2'
    REDUCED_KAMATZ = '\u05B3'
    HIRIK = '\u05B4'
    TZEIRE = '\u05B5'
    SEGOL = '\u05B6'
    PATAKH = '\u05B7'
    KAMATZ = '\u05B8'
    HOLAM = '\u05B9'
    KUBUTZ = '\u05BB'
    SHURUK = '\u05BC'
    METEG = '\u05BD'


HEBREW_LETTERS = [chr(c) for c in range(0x05d0, 0x05ea + 1)]

NIQQUD = [RAFE] + [chr(c) for c in range(0x05b0, 0x05bc + 1)] + ['\u05b7']

HOLAM = Niqqud.HOLAM

SHIN_YEMANIT = '\u05c1'
SHIN_SMALIT = '\u05c2'
NIQQUD_SIN = [RAFE, SHIN_YEMANIT, SHIN_SMALIT]  # RAFE is for acronyms

DAGESH_LETTER = '\u05bc'
DAGESH = [RAFE, DAGESH_LETTER]  # note that DAGESH and SHURUK are one and the same

ANY_NIQQUD = [RAFE] + NIQQUD[1:] + NIQQUD_SIN[1:] + DAGESH[1:]

VALID_LETTERS = [' ', '!', '"', "'", '(', ')', ',', '-', '.', ':', ';', '?'] + HEBREW_LETTERS
SPECIAL_TOKENS = ['H', 'O', '5']

ENDINGS_TO_REGULAR = dict(zip('ךםןףץ', 'כמנפצ'))


def normalize(c):
    if c in VALID_LETTERS: return c
    if c in ENDINGS_TO_REGULAR: return ENDINGS_TO_REGULAR[c]
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
    normalized: str
    dagesh: str
    sin: str
    niqqud: str

    def __str__(self):
        return self.letter + self.dagesh + self.sin + self.niqqud

    def __repr__(self):
        return repr((self.letter, bool(self.dagesh), bool(self.sin), ord(self.niqqud or chr(0))))

    def vocalize(self):
        return self._replace(niqqud=vocalize_niqqud(self.niqqud),
                             sin=self.sin.replace(RAFE, ''),
                             dagesh=vocalize_dagesh(self.letter, self.dagesh))


def items_to_text(items: List[HebrewItem]) -> str:
    return ''.join(str(item) for item in items).replace(RAFE, '')


def vocalize_dagesh(letter, dagesh):
    if letter not in 'בכפ':
        return ''
    return dagesh.replace(RAFE, '')


def vocalize_niqqud(c):
    # FIX: HOLAM / KUBBUTZ cannot be handled here correctly
    if c in [Niqqud.KAMATZ, Niqqud.PATAKH, Niqqud.REDUCED_PATAKH]:
        return Niqqud.PATAKH

    if c in [Niqqud.HOLAM, Niqqud.REDUCED_KAMATZ]:
        return Niqqud.HOLAM  # TODO: Kamatz-katan

    if c in [Niqqud.SHURUK, Niqqud.KUBUTZ]:
        return Niqqud.KUBUTZ

    if c in [Niqqud.TZEIRE, Niqqud.SEGOL, Niqqud.REDUCED_SEGOL]:
        return Niqqud.SEGOL

    if c == Niqqud.SHVA:
        return ''

    return c.replace(RAFE, '')


def is_hebrew_letter(letter: str) -> bool:
    return '\u05d0' <= letter <= '\u05ea'


def can_dagesh(letter):
    return letter in ('בגדהוזטיכלמנספצקשת' + 'ךף')


def can_sin(letter):
    return letter == 'ש'


def can_niqqud(letter):
    return letter in ('אבגדהוזחטיכלמנסעפצקרשת' + 'ךן')


def can_any(letter):
    return can_niqqud(letter) or can_dagesh(letter) or can_sin(letter)


def iterate_dotted_text(text: str) -> Iterator[HebrewItem]:
    n = len(text)
    text += '  '
    i = 0
    while i < n:
        letter = text[i]

        dagesh = RAFE if can_dagesh(letter) else ''
        sin = RAFE if can_sin(letter) else ''
        niqqud = RAFE if can_niqqud(letter) else ''
        normalized = normalize(letter)
        i += 1
        nbrd = text[i - 15:i + 15].split()[1:-1]

        assert letter not in ANY_NIQQUD, f'{i}, {nbrd}, {letter}, {[name_of(c) for word in nbrd for c in word]}'

        if is_hebrew_letter(normalized):
            if text[i] == DAGESH_LETTER:
                # assert dagesh == RAFE, (text[i-5:i+5])
                dagesh = text[i]
                i += 1
            if text[i] in NIQQUD_SIN:
                # assert sin == RAFE, (text[i-5:i+5])
                sin = text[i]
                i += 1
            if text[i] in NIQQUD:
                # assert niqqud == RAFE, (text[i-5:i+5])
                niqqud = text[i]
                i += 1
            if letter == 'ו' and dagesh == DAGESH_LETTER and niqqud == RAFE:
                dagesh = RAFE
                niqqud = DAGESH_LETTER

        yield HebrewItem(letter, normalized, dagesh, sin, niqqud)


def iterate_file(path):
    with open(path, encoding='utf-8') as f:
        text = ''.join(s + ' ' for s in f.read().split())
        try:
            yield from iterate_dotted_text(text)
        except AssertionError as ex:
            ex.args += (path,)
            raise


def is_space(c):
    if isinstance(c, HebrewItem):
        return c.letter == ' '
    elif isinstance(c, str):
        return c == ' '
    assert False


def split_by_length(characters: Iterable, maxlen: int):
    assert maxlen > 1
    out = []
    space = maxlen
    for c in characters:
        if is_space(c):
            space = len(out)
        out.append(c)
        if len(out) == maxlen - 1:
            yield out[:space+1]
            out = out[space+1:]
    if out:
        yield out


@dataclass(frozen=True)
class Token:
    items: tuple[HebrewItem]

    def __str__(self):
        return ''.join(str(c) for c in self.items)

    def __lt__(self, other: 'Token'):
        return (self.to_undotted(), str(self)) < (other.to_undotted(), str(other))

    def split_on_hebrew(self) -> tuple[str, 'Token', str]:
        start = 0
        end = len(self.items) - 1
        while True:
            if start >= len(self.items):
                return ('', Token(()), '')
            if self.items[start].letter in HEBREW_LETTERS + ANY_NIQQUD:
                break
            start += 1
        while self.items[end].letter not in HEBREW_LETTERS + ANY_NIQQUD:
            end -= 1
        return (''.join(c.letter for c in self.items[:start]),
                Token(self.items[start:end+1]),
                ''.join(c.letter for c in self.items[end+1:]))

    def __bool__(self):
        return bool(self.items)

    def __eq__(self, other):
        return self.items == other.items

    @lru_cache()
    def to_undotted(self):
        return ''.join(str(c.letter) for c in self.items)

    def is_undotted(self):
        return len(self.items) > 1 and all(c.niqqud in [RAFE, ''] for c in self.items)

    def is_definite(self):
        return len(self.items) > 2 and self.items[0].niqqud == 'הַ'[-1] and self.items[0].letter in 'כבלה'

    def vocalize(self) -> 'Token':
        return Token(tuple([c.vocalize() for c in self.items]))


def tokenize_into(tokens_list: List[Token], char_iterator: Iterator[HebrewItem], strip_nonhebrew: bool) -> Iterator[HebrewItem]:
    current = []
    for c in char_iterator:
        if c.letter.isspace() or c.letter == '-':
            if current:
                token = Token(tuple(current))
                if strip_nonhebrew:
                    _, token, _ = token.split_on_hebrew()
                tokens_list.append(token)
            current = []
        else:
            current.append(c)
        yield c
    if current:
        token = Token(tuple(current))
        if strip_nonhebrew:
            _, token, _ = token.split_on_hebrew()
        tokens_list.append(token)


def tokenize(iterator: Iterator[HebrewItem], strip_nonhebrew=False) -> List[Token]:
    tokens = []
    _ = list(tokenize_into(tokens, iterator, strip_nonhebrew))
    return tokens


def collect_wordmap(tokens: Iterable[Token]):
    word_dict = defaultdict(Counter)
    for token in tokens:
        word_dict[token.to_undotted()][str(token)] += 1
    return word_dict


def collect_tokens(paths: Iterable[str]):
    return tokenize(itertools.chain.from_iterable(iterate_file(path) for path in utils.iterate_files(paths)),
                    strip_nonhebrew=True)


def stuff(tokens):
    stripped_tokens = [token.strip_nonhebrew() for token in tokens if token.strip_nonhebrew()]
    word_dict = collect_wordmap(stripped_tokens)
    # for k, v in sorted(word_dict.items(), key=lambda kv: (len(kv[1]), sum(kv[1].values()))):
    #     print(k, ':', str(v).replace('Counter', ''))
    # print(len(word_dict))

    for t in tokens:
        if t.is_definite() and t.items[1].letter not in 'אהחער' and not t.items[1].dagesh:
            print(t)
    #
    # for t in stripped_tokens:
    #     if t.is_undotted() and '"' not in t.to_undotted():
    #         print(t)
    # for k, v in word_dict.items():
    #     if "וו" in k:
    #         print(v)


def remove_niqqud(text: str) -> str:
    return re.sub('[\u05B0-\u05BC\u05C1\u05C2ׇ\u05c7]', '', text)


def average_wordlen(path):
    token_lens = [len(t.strip_nonhebrew().items) for t in collect_tokens([path])
                  if len(t.strip_nonhebrew().items) > 1]
    import numpy as np
    return np.mean(token_lens)


def name_of(c):
    if 'א' <= c <= 'ת':
        return c
    if c == DAGESH_LETTER: return 'דגש\שורוק'
    if c == Niqqud.KAMATZ: return 'קמץ'
    if c == Niqqud.PATAKH: return 'פתח'
    if c == Niqqud.TZEIRE: return 'צירה'
    if c == Niqqud.SEGOL: return 'סגול'
    if c == Niqqud.SHVA: return 'שוא'
    if c == Niqqud.HOLAM: return 'חולם'
    if c == Niqqud.KUBUTZ: return 'קובוץ'
    if c == Niqqud.HIRIK: return 'חיריק'
    if c == Niqqud.REDUCED_KAMATZ: return 'חטף-קמץ'
    if c == Niqqud.REDUCED_PATAKH: return 'חטף-פתח'
    if c == Niqqud.REDUCED_SEGOL: return 'חטף-סגול'
    if c == SHIN_SMALIT: return 'שין-שמאלית'
    if c == SHIN_YEMANIT: return 'שין-ימנית'
    if c.isprintable():
        return c
    return "לא ידוע ({})".format(hex(ord(c)))


def find_longest_undotted(text):
    text = re.sub(r'[^\u0590-\u05f4#]', '', text)
    return max(re.findall(r'[\u05d0-\u05ea]+', text), key=len)


def print_longest_undotted_files(path):
    undotted_list = [(find_longest_undotted(utils.read_file(filename)), filename) for filename in utils.iterate_files(path)]
    undotted_list.sort(key=lambda x: len(x[1]))
    for text, filename in undotted_list:
        if len(text) > 25:
            print(filename)


if __name__ == '__main__':
    # print_longest_undotted_files(['../gender_dots/scraping/scrape_data/Dicta'])
    tokens = collect_tokens(['tests/test/expected'])
    # stuff(tokens)
    print(len(tokens))
    # 
    # for i, t in enumerate(tokens):
    #     vv = 'פניה'
    #     if vv == remove_niqqud(str(t)):
    #         print(' '.join(str(x).replace(RAFE, '') for x in tokens[i-1:i+2]))
