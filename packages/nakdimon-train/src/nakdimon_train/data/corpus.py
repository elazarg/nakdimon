"""Corpus reading and encoding on the v2 text layer (torch-free).

Reads dotted corpus files with the order-tolerant parser (legacy and NFC files
alike), applies the v2 mark folds, cuts char-aligned training windows, and
encodes with a spec inventory into numpy arrays.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from nakdimon import hebrew
from nakdimon.encode import Inventory
from nakdimon.spec import load_spec

QAMATS = "ָ"
QAMATS_QATAN = "ׇ"

# Default corpus mixture, carried over from v1's Full recipe.
CORPUS_GROUPS: dict[str, tuple[str, ...]] = {
    "premodern": (
        "hebrew_diacritized/poetry",
        "hebrew_diacritized/rabanit",
        "hebrew_diacritized/pre_modern",
        "hebrew_diacritized/shortstoryproject_predotted",
    ),
    "automatic": ("hebrew_diacritized/shortstoryproject_Dicta",),
    "modern": (
        "hebrew_diacritized/modern",
        "hebrew_diacritized/dictaTestCorpus",
    ),
}
VALIDATION_PATH = "hebrew_diacritized/validation/modern"


def iter_files(paths: tuple[str, ...] | list[str]) -> Iterator[Path]:
    for base in paths:
        p = Path(base)
        if p.is_dir():
            yield from sorted(q for q in p.rglob("*") if q.is_file())
        elif p.exists():
            yield p


def _fold_niqqud(niqqud: str, folds: dict[str, str], fold_qq: bool) -> str:
    if fold_qq and niqqud == QAMATS_QATAN:
        return QAMATS
    return folds.get(niqqud, niqqud)


def read_dotted_file(path: Path, *, fold_qq: bool = True) -> list[hebrew.HebrewItem]:
    """v1 file-reading semantics (whitespace-collapse, trailing space) so window
    contents line up with what v1 trained on; marks folded per the v2 inventory."""
    text = "".join(word + " " for word in path.read_text(encoding="utf-8").split())
    folds = load_spec()["inventories"]["v2"].get("data_folds", {})
    items = hebrew.decompose(hebrew.clean_text(text))
    return [item._replace(niqqud=_fold_niqqud(item.niqqud, folds, fold_qq)) for item in items]


def windows_of(items: list[hebrew.HebrewItem], window: int) -> Iterator[list[hebrew.HebrewItem]]:
    for start in range(0, len(items), window):
        yield items[start : start + window]


def encode_corpus(
    paths: tuple[str, ...] | list[str],
    window: int,
    inventory: Inventory,
    *,
    fold_qq: bool = True,
) -> dict[str, np.ndarray]:
    """All windows of all files under `paths`, encoded and zero-padded to
    `window`. Keys: letters, niqqud, dagesh, sin — each [N, window] int32."""
    rows: dict[str, list[np.ndarray]] = {"letters": [], "niqqud": [], "dagesh": [], "sin": []}
    for path in iter_files(paths):
        items = read_dotted_file(path, fold_qq=fold_qq)
        if not items:
            continue
        for win in windows_of(items, window):
            encoded = inventory.encode_items(win)
            for key in rows:
                row = np.zeros(window, dtype=np.int32)
                row[: len(win)] = encoded[key]
                rows[key].append(row)
    if not rows["letters"]:
        raise ValueError(f"no corpus content under {paths}")
    return {key: np.stack(value) for key, value in rows.items()}


def load_inventory(name: str = "v2") -> Inventory:
    return Inventory(load_spec()["inventories"][name])
