"""v2 port of the Nakdimon evaluation metrics (DEC/CHA/WOR/VOC + OOV variants).

Faithful reimplementation of nakdimon/metrics.py + the MajorityDiacritizer half of
nakdimon/external_apis.py, rebuilt on the v2 text layer (`nakdimon.hebrew`) and the
v2 token layer (`nakdimon_train.tokens`). Must reproduce spec/golden/metrics_v1.json
exactly on the legacy-ordered files under tests/new -- see test_metrics_golden.py.

No torch import: this module (and tokens.py) must be usable without the training
extra, so eval.py can run under just the base `nakdimon` (numpy + onnxruntime) stack.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from nakdimon import hebrew

from nakdimon_train.tokens import HEBREW_LETTERS, Token, decompose_legacy, split_nonhebrew, tokenize, vocalize

# Fold niqqud labels that don't exist in the v1 inventory (and therefore don't exist in
# any v1-era gold text) onto their v1 equivalents, so v2-model output -- which may emit
# qamats qatan, and in principle the v1 dead class U+05BA -- compares fairly against
# legacy gold. No-op on legacy corpora, which never contain these codepoints; see
# DESIGN.md section 4 ("metrics are reported both raw and qq-folded").
NIQQUD_FOLD = {
    "ׇ": "ָ",  # qamats qatan -> qamats
    "ֺ": "ֹ",  # holam haser for vav -> holam
}


def fold_item(item: hebrew.HebrewItem) -> hebrew.HebrewItem:
    folded = NIQQUD_FOLD.get(item.niqqud)
    if folded is None:
        return item
    return item._replace(niqqud=folded)


@dataclass(frozen=True)
class Document:
    name: str
    system: str
    text: str
    fold: bool = True
    legacy: bool = False

    def hebrew_items(self) -> list[hebrew.HebrewItem]:
        # Default parser is v2's order-tolerant hebrew.decompose, which makes every
        # metric invariant under NFC normalization of the input files. legacy=True
        # swaps in v1's positional grammar (tokens.decompose_legacy) -- needed only
        # to reproduce spec/golden/metrics_v1.json bit-for-bit on the pre-NFC tree,
        # where a "double dagesh" glitch in some Nakdimon test outputs parses
        # differently (see decompose_legacy's docstring).
        items = decompose_legacy(self.text) if self.legacy else hebrew.decompose(self.text)
        if self.fold:
            items = [fold_item(item) for item in items]
        return items

    def tokens(self) -> list[Token]:
        return tokenize(self.hebrew_items(), strip_nonhebrew=False)

    def vocalized_tokens(self) -> list[Token]:
        return [vocalize(t) for t in self.tokens()]


@dataclass(frozen=True)
class DocumentPack:
    source: str
    name: str
    docs: dict[str, Document]

    def __getitem__(self, item: str) -> Document:
        return self.docs[item]

    @property
    def expected(self) -> Document:
        return self.docs["expected"]

    @property
    def actual(self) -> Document:
        assert len(self.docs) == 2
        assert "expected" in self.docs
        return self.docs[(set(self.docs.keys()) - {"expected"}).pop()]


def system_path_from_expected(path_to_expected: Path, system: str) -> Path:
    # v1 does a plain (all-occurrences) string replace, not a single path-segment
    # swap; matched here for parity (irrelevant in practice since 'expected' occurs
    # exactly once in these test-set paths).
    return Path(str(path_to_expected).replace("expected", system))


def read_document(system: str, path: Path, *, fold: bool = True, legacy: bool = False) -> Document:
    text = " ".join(path.read_text(encoding="utf8").strip().split())
    return Document(path.name, system, text, fold=fold, legacy=legacy)


def read_document_pack(
    path_to_expected: Path, systems: list[str], *, fold: bool = True, legacy: bool = False
) -> DocumentPack:
    return DocumentPack(
        path_to_expected.parent.name,
        path_to_expected.name,
        {
            system: read_document(
                system, system_path_from_expected(path_to_expected, system), fold=fold, legacy=legacy
            )
            for system in systems
        },
    )


@dataclass
class Stats:
    basepath: Path
    vocabulary: MajorityVocabulary | None = None
    fold: bool = True
    legacy: bool = False

    def _folders(self) -> list[Path]:
        # Default: sorted, so aggregation is deterministic across filesystems.
        # legacy=True keeps Path.iterdir() order -- v1's aggregation order, which the
        # v1 golden is pinned to (numpy mean is order-sensitive at the 1e-9 level).
        return list(self.basepath.iterdir()) if self.legacy else sorted(self.basepath.iterdir())

    def _folder_docs(self, folder: Path) -> list[Path]:
        return list(folder.iterdir()) if self.legacy else sorted(folder.iterdir())

    def iter_documents(self, systems: list[str]) -> Iterator[DocumentPack]:
        for folder in self._folders():
            for path_to_expected in self._folder_docs(folder):
                yield read_document_pack(path_to_expected, systems, fold=self.fold, legacy=self.legacy)

    def iter_documents_by_folder(self, systems: list[str]) -> Iterator[list[DocumentPack]]:
        for folder in self._folders():
            yield [
                read_document_pack(path_to_expected, systems, fold=self.fold, legacy=self.legacy)
                for path_to_expected in self._folder_docs(folder)
            ]

    def is_oov(self, word: str) -> bool:
        if self.vocabulary is None:
            return True
        return self.vocabulary.is_oov(word)

    def metric_cha(self, doc_pack: DocumentPack) -> float:
        """Character-level agreement between actual and expected, over items that
        can carry any diacritic (dagesh, sin, or niqqud).
        """
        actual_items = doc_pack.actual.hebrew_items()
        expected_items = doc_pack.expected.hebrew_items()
        # strict=False (truncate to the shorter sequence) is load-bearing: actual and
        # expected item streams can differ in length (e.g. a system that drops or adds
        # a token), and v1 silently truncates rather than erroring. Do not "fix" this.
        return self.mean_equal(
            (x, y)
            for x, y in zip(actual_items, expected_items, strict=False)
            if hebrew.can_dagesh(x.letter) or hebrew.can_sin(x.letter) or hebrew.can_niqqud(x.letter)
        )

    def metric_dec(self, doc_pack: DocumentPack) -> float:
        """Nontrivial-decision agreement: pools the niqqud/dagesh/sin channels (each
        filtered to items where that channel applies) into one mean.
        """
        actual_items = doc_pack.actual.hebrew_items()
        expected_items = doc_pack.expected.hebrew_items()
        return self.mean_equal(
            (
                (x.niqqud, y.niqqud)
                for x, y in zip(actual_items, expected_items, strict=False)
                if hebrew.can_niqqud(x.letter)
            ),
            (
                (x.dagesh, y.dagesh)
                for x, y in zip(actual_items, expected_items, strict=False)
                if hebrew.can_dagesh(x.letter)
            ),
            (
                (x.sin, y.sin)
                for x, y in zip(actual_items, expected_items, strict=False)
                if hebrew.can_sin(x.letter)
            ),
        )

    def metric_wor(self, doc_pack: DocumentPack, oov: bool = False) -> float:
        """Token-level agreement, for tokens with more than one Hebrew letter."""
        return self.mean_equal(
            (x, y)
            for x, y in zip(doc_pack.actual.tokens(), doc_pack.expected.tokens(), strict=False)
            if x.is_hebrew() and (not oov or self.is_oov(x.to_text()))
        )

    def metric_voc(self, doc_pack: DocumentPack, oov: bool = False) -> float:
        """Token-level agreement over vocalization (coarse vowel classes)."""
        return self.mean_equal(
            (x, y)
            for x, y in zip(doc_pack.actual.vocalized_tokens(), doc_pack.expected.vocalized_tokens(), strict=False)
            if x.is_hebrew() and (not oov or self.is_oov(x.to_text()))
        )

    def mean_equal(self, *pair_iterables: Iterable[tuple]) -> float:
        total = 0
        acc = 0
        for pair_iterable in pair_iterables:
            pairs = list(pair_iterable)
            total += len(pairs)
            acc += sum(x == y for x, y in pairs)
        if not total:
            return 0.0
        return acc / total

    def all_metrics(self, doc_pack: DocumentPack) -> dict[str, float]:
        return {
            "dec": self.metric_dec(doc_pack),
            "cha": self.metric_cha(doc_pack),
            "wor": self.metric_wor(doc_pack),
            "voc": self.metric_voc(doc_pack),
            "wor_oov": self.metric_wor(doc_pack, oov=True),
            "voc_oov": self.metric_voc(doc_pack, oov=True),
        }

    def metricwise_mean(self, iterable: Iterable[dict[str, float]]) -> dict[str, float]:
        items = list(iterable)
        keys = items[0].keys()
        return {key: float(np.mean([item[key] for item in items])) for key in keys}

    def macro_average(self, system: str) -> dict[str, float]:
        """Mean over folders of (mean over documents within that folder)."""
        return self.metricwise_mean(
            self.metricwise_mean(self.all_metrics(doc_pack) for doc_pack in folder_packs)
            for folder_packs in self.iter_documents_by_folder(["expected", system])
        )

    def micro_average(self, system: str) -> dict[str, float]:
        return self.metricwise_mean(self.all_metrics(doc_pack) for doc_pack in self.iter_documents(["expected", system]))


def _iterate_files(paths: Iterable[Path]) -> Iterator[Path]:
    """Port of v1 utils.iterate_files: yield `path` itself if it's a file, else walk
    the directory tree and yield every file under it (os.walk order).
    """
    for base in paths:
        if not base.is_dir():
            yield base
            continue
        for root, _dirs, files in os.walk(base):
            for fname in files:
                yield Path(root) / fname


def _iterate_file_items(path: Path, *, legacy: bool = False) -> Iterator[hebrew.HebrewItem]:
    # Port of v1 hebrew.iterate_file: collapse all whitespace, and re-join with a
    # single trailing space per token so tokenize() sees a clean boundary at both
    # word breaks and file boundaries (letting callers chain items across files).
    text = "".join(word + " " for word in path.read_text(encoding="utf-8").split())
    yield from (decompose_legacy(text) if legacy else hebrew.decompose(text))


@dataclass
class MajorityVocabulary:
    """Port of v1 external_apis.MajorityDiacritizer, restricted to what the metrics
    need: the OOV vocabulary (`is_oov`). `diacritize_token`/`__call__` are ported for
    completeness (a majority baseline needs them to actually diacritize text) but are
    not exercised by the golden test, which only checks `is_oov`.
    """

    dictionary: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_paths(cls, paths: Iterable[Path], *, legacy: bool = False) -> MajorityVocabulary:
        possibilities: defaultdict[str, Counter[str]] = defaultdict(Counter)
        item_stream = (item for path in _iterate_files(paths) for item in _iterate_file_items(path, legacy=legacy))
        for token in tokenize(item_stream, strip_nonhebrew=True):
            word = token.to_text()
            if not word:
                continue
            _, word, _ = split_nonhebrew(word)
            if not word:
                continue
            possibilities[hebrew.remove_niqqud(word)][word] += 1
            for item in token.items:
                if item.letter in HEBREW_LETTERS:
                    possibilities[item.letter][hebrew.items_to_text([item])] += 1
        dictionary = {key: options.most_common(1)[0][0] for key, options in possibilities.items()}
        return cls(dictionary)

    def is_oov(self, word: str) -> bool:
        _, inner, _ = split_nonhebrew(hebrew.remove_niqqud(word))
        return inner not in self.dictionary

    def diacritize_token(self, word: str) -> str:
        left, word, right = split_nonhebrew(word)
        word = hebrew.remove_niqqud(word)
        result = self.dictionary.get(word, word)
        return left + result + right

    def __call__(self, text: str) -> str:
        return " ".join(self.diacritize_token(token) for token in hebrew.remove_niqqud(text).split())
