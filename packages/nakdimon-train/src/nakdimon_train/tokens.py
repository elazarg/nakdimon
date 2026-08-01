"""Token-level Hebrew text utilities for the metrics port.

Faithful reimplementation of the token semantics from the legacy
``nakdimon.hebrew`` module (``Token``, ``tokenize``, ``vocalize``,
``split_on_hebrew``, ``split_nonhebrew``), rebuilt on top of the v2
``nakdimon.hebrew.decompose()`` item representation (see that module's
``HebrewItem`` / ``can_dagesh`` / ``can_sin`` / ``can_niqqud`` / ``items_to_text``).

v1 serialized a ``HebrewItem`` as ``letter + dagesh + sin + niqqud``; v2's
``items_to_text`` emits NFC canonical order (``letter + niqqud + dagesh + sin``).
Metric equality never depends on serialization order because both sides of every
comparison go through the *same* fixed order (either raw item-tuple equality, or a
string built the same way on both sides) -- see metrics.py.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from nakdimon.hebrew import RAFE, HebrewItem, can_dagesh, can_niqqud, can_sin, items_to_text, normalize_char

HEBREW_LETTERS = frozenset(chr(c) for c in range(0x05D0, 0x05EA + 1))

# Niqqud codepoints, named as in the legacy nakdimon.hebrew.Niqqud class.
SHVA = "ְ"
REDUCED_SEGOL = "ֱ"
REDUCED_PATAKH = "ֲ"
REDUCED_KAMATZ = "ֳ"
HIRIK = "ִ"
TZEIRE = "ֵ"
SEGOL = "ֶ"
PATAKH = "ַ"
KAMATZ = "ָ"
HOLAM = "ֹ"
KUBUTZ = "ֻ"
SHURUK = "ּ"

SHIN_YEMANIT = "ׁ"
SHIN_SMALIT = "ׂ"
DAGESH_LETTER = "ּ"

# Any mark that can attach to a Hebrew letter: RAFE + the niqqud range U+05B0-U+05BC
# (which subsumes the dagesh/shuruk mark U+05BC) + the shin/sin dots. Mirrors legacy
# ANY_NIQQUD exactly. This is used only to recognize stray/isolated marks (e.g. a
# combining mark with no preceding base letter in malformed input) as "Hebrew-ish" for
# token-boundary trimming, same as v1 -- ordinary parsed items never have a mark as
# their own `.letter`.
_ANY_MARK = frozenset([RAFE, SHIN_YEMANIT, SHIN_SMALIT] + [chr(c) for c in range(0x05B0, 0x05BC + 1)])
_HEBREW_OR_MARK = HEBREW_LETTERS | _ANY_MARK

# Letters that can carry a "real" dagesh (as opposed to a rafe-only slot): beged-kefet.
_DAGESH_LETTERS = "בכפ"

# Fixed-order "slots" mirroring v1's iterate_dotted_text grammar: dagesh-slot, then
# sin-slot, then niqqud-slot, each consuming at most one character if it belongs to
# that slot's set. NOT order-tolerant / last-wins like v2's hebrew.decompose.
_SIN_SLOT = frozenset([RAFE, SHIN_YEMANIT, SHIN_SMALIT])
# v1's NIQQUD set runs 0x05B0..0x05BC *inclusive* -- it accidentally includes the
# dagesh/shuruk mark U+05BC alongside the "true" niqqud range 0x05B0..0x05BB. That
# off-by-one is the source of the one trap decompose_legacy exists to reproduce.
_NIQQUD_SLOT = frozenset([RAFE] + [chr(c) for c in range(0x05B0, 0x05BC + 1)])


def decompose_legacy(text: str) -> list[HebrewItem]:
    """Sequential, fixed-slot-order decomposition: a byte-exact port of v1
    nakdimon.hebrew.iterate_dotted_text, built on v2's can_dagesh/can_sin/can_niqqud/
    normalize_char/RAFE/HebrewItem primitives -- used instead of v2's
    `nakdimon.hebrew.decompose` for metrics (see metrics.Document) because the two
    are NOT equivalent on one real, reproducible artifact.

    v2's decompose() is order-tolerant: marks after a letter may appear in any order,
    last-wins per channel, and U+05BC (the dagesh/shuruk mark) always routes to the
    dagesh channel. v1's grammar is positional: dagesh-slot, then sin-slot, then
    niqqud-slot, each greedily consuming one character of the right class. Because
    v1's "niqqud" character class runs 0x05B0..0x05BC *inclusive* -- it accidentally
    includes the dagesh/shuruk mark too -- a run of *two* consecutive U+05BC marks
    after a letter (e.g. a literal "וּּ", a "double dagesh" glitch that
    occurs for real in tests/new/Nakdimon*/{nrg/6,kol/8,books/6,eureka/1,...}.txt) has
    its first U+05BC consumed by the dagesh-slot and its second consumed by the
    niqqud-slot, yielding the nonsensical but real HebrewItem with dagesh == niqqud ==
    U+05BC. v2's decompose() cannot produce that item under any input (U+05BC never
    reaches the niqqud channel by construction), so exact reproduction of
    spec/golden/metrics_v1.json -- captured from v1's output on these exact files --
    requires v1's exact grammar here, not v2's (deliberately fixed) one. On
    well-formed legacy-ordered text without this glitch, the two agree.
    """
    items: list[HebrewItem] = []
    n = len(text)
    padded = text + "  "  # lookahead room, as in v1
    i = 0
    while i < n:
        letter = padded[i]
        dagesh = RAFE if can_dagesh(letter) else ""
        sin = RAFE if can_sin(letter) else ""
        niqqud = RAFE if can_niqqud(letter) else ""
        normalized = normalize_char(letter)
        i += 1

        if letter in HEBREW_LETTERS:
            if padded[i] == DAGESH_LETTER:
                dagesh = padded[i]
                i += 1
            if padded[i] in _SIN_SLOT:
                sin = padded[i]
                i += 1
            if padded[i] in _NIQQUD_SLOT:
                niqqud = padded[i]
                i += 1
            # A dagesh on vav with no explicit niqqud is really the shuruk vowel --
            # but only fires if the niqqud-slot came up empty (RAFE); the double-dagesh
            # glitch above leaves niqqud == U+05BC, not RAFE, so it does NOT fire then.
            if letter == "ו" and dagesh == DAGESH_LETTER and niqqud == RAFE:
                dagesh = RAFE
                niqqud = DAGESH_LETTER

        items.append(HebrewItem(letter, normalized, dagesh, sin, niqqud))
    return items


@dataclass(frozen=True)
class Token:
    """A run of HebrewItems between whitespace/hyphen boundaries. Equality and
    hashing are by `items` (structural), matching v1's Token.__eq__.
    """

    items: tuple[HebrewItem, ...]

    def __bool__(self) -> bool:
        return bool(self.items)

    def to_undotted(self) -> str:
        return "".join(item.letter for item in self.items)

    def to_text(self) -> str:
        return items_to_text(list(self.items))

    def is_hebrew(self) -> bool:
        """True iff the token contains more than one Hebrew letter (v1's threshold
        for "worth scoring at the word level" -- excludes single letters/punctuation).
        """
        return sum(1 for item in self.items if item.letter in HEBREW_LETTERS) > 1

    def split_on_hebrew(self) -> tuple[str, Token, str]:
        """Trim leading/trailing items that are neither a Hebrew letter nor a mark,
        returning (left_junk, trimmed_token, right_junk). If no item qualifies, returns
        ('', Token(()), '') -- the empty-token case, ported verbatim from v1.
        """
        start = 0
        while True:
            if start >= len(self.items):
                return "", Token(()), ""
            if self.items[start].letter in _HEBREW_OR_MARK:
                break
            start += 1
        end = len(self.items) - 1
        while self.items[end].letter not in _HEBREW_OR_MARK:
            end -= 1
        pre = "".join(item.letter for item in self.items[:start])
        post = "".join(item.letter for item in self.items[end + 1 :])
        return pre, Token(self.items[start : end + 1]), post


def tokenize(items: Iterable[HebrewItem], strip_nonhebrew: bool) -> list[Token]:
    """Split `items` into Tokens on whitespace and hyphens (v1: `letter.isspace() or
    letter == '-'`). When `strip_nonhebrew`, each token is additionally trimmed via
    `Token.split_on_hebrew` (which can yield an empty Token(()) -- callers that build a
    vocabulary must skip those, as v1 does via `if word:`).
    """
    result: list[Token] = []
    current: list[HebrewItem] = []
    for item in items:
        if item.letter.isspace() or item.letter == "-":
            if current:
                token = Token(tuple(current))
                if strip_nonhebrew:
                    _, token, _ = token.split_on_hebrew()
                result.append(token)
            current = []
        else:
            current.append(item)
    if current:
        token = Token(tuple(current))
        if strip_nonhebrew:
            _, token, _ = token.split_on_hebrew()
        result.append(token)
    return result


def _vocalize_niqqud(c: str) -> str:
    # FIX (inherited from v1): HOLAM / KUBUTZ collapse loses the shuruk/holam-vav
    # distinction; not fixable at this per-item level without lookahead.
    if c in (KAMATZ, PATAKH, REDUCED_PATAKH):
        return PATAKH
    if c in (HOLAM, REDUCED_KAMATZ):
        return HOLAM
    if c in (SHURUK, KUBUTZ):
        return KUBUTZ
    if c in (TZEIRE, SEGOL, REDUCED_SEGOL):
        return SEGOL
    if c == SHVA:
        return ""
    return c.replace(RAFE, "")


def _vocalize_dagesh(letter: str, dagesh: str) -> str:
    if letter not in _DAGESH_LETTERS:
        return ""
    return dagesh.replace(RAFE, "")


def vocalize_item(item: HebrewItem) -> HebrewItem:
    """Collapse a HebrewItem to its "vocalization" -- the coarse vowel class used by
    the VOC metric, discarding sin-dot rafe and dagesh outside beged-kefet letters.
    """
    return item._replace(
        niqqud=_vocalize_niqqud(item.niqqud),
        sin=item.sin.replace(RAFE, ""),
        dagesh=_vocalize_dagesh(item.letter, item.dagesh),
    )


def vocalize(token: Token) -> Token:
    return Token(tuple(vocalize_item(item) for item in token.items))


def lsplit_nonhebrew(word: str) -> tuple[str, str]:
    """Split off a leading run of non-Hebrew characters. `word` must be nonempty.

    Trap ported verbatim from v1: if no character matches, the `for` loop still
    leaves `i == len(word) - 1` (not `len(word)`), so the split point lands one
    character before the end rather than declaring the whole word non-Hebrew. In
    practice this path is unreachable from metrics.py's actual call sites (both
    `is_oov` and vocabulary building only ever call this on words already known to
    contain a Hebrew letter), but a "fix" here would silently diverge from v1 on any
    input where it *does* trigger, so it is kept as-is.
    """
    assert word
    i = 0
    for i in range(len(word)):
        if word[i] in HEBREW_LETTERS or word[i] in _ANY_MARK:
            break
    return word[:i], word[i:]


def rsplit_nonhebrew(word: str) -> tuple[str, str]:
    assert word
    right, reversed_word = lsplit_nonhebrew(word[::-1])
    return reversed_word[::-1], right[::-1]


def split_nonhebrew(word: str) -> tuple[str, str, str]:
    assert word
    left, word = lsplit_nonhebrew(word)
    word, right = rsplit_nonhebrew(word)
    return left, word, right
