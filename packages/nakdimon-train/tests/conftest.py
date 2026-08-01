"""Test bootstrap for the nakdimon-train metrics port.

This repo's root venv has the LEGACY v1 `nakdimon` package installed editable (from
the repo-root `nakdimon/` package, name "nakdimon" version 0.2.1) -- that's what
`uv run` at the repo root gives you by default. Without fixing up sys.path first,
`import nakdimon` from these tests would silently resolve to v1 instead of the v2
port (`packages/nakdimon/src/nakdimon`) these tests are meant to exercise. So: put
the v2 `nakdimon` package and this package's own `nakdimon_train` on sys.path ahead
of everything else, and evict any legacy `nakdimon` module already cached in
sys.modules from an earlier, unpatched import.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
V2_NAKDIMON_SRC = REPO_ROOT / "packages" / "nakdimon" / "src"
TRAIN_SRC = REPO_ROOT / "packages" / "nakdimon-train" / "src"

for _path in (TRAIN_SRC, V2_NAKDIMON_SRC):
    _entry = str(_path)
    if _entry in sys.path:
        sys.path.remove(_entry)
    sys.path.insert(0, _entry)

_stale_nakdimon = sys.modules.get("nakdimon")
if _stale_nakdimon is not None and str(V2_NAKDIMON_SRC) not in (getattr(_stale_nakdimon, "__file__", None) or ""):
    for _name in [m for m in sys.modules if m == "nakdimon" or m.startswith("nakdimon.")]:
        del sys.modules[_name]

from nakdimon_train.metrics import MajorityVocabulary, Stats

GOLDEN_V1_PATH = REPO_ROOT / "spec" / "golden" / "metrics_v1.json"
GOLDEN_V2_PATH = REPO_ROOT / "spec" / "golden" / "metrics_v2.json"
TEST_SET_DIR = REPO_ROOT / "tests" / "new"
VOCABULARY_DIR = REPO_ROOT / "hebrew_diacritized" / "modern"


def tree_is_pre_nfc() -> bool:
    """True while the test-set files still carry v1 (non-NFC) mark order. The v1
    golden can only be reproduced on that tree: decompose_legacy is a positional
    parser and mis-reads NFC-ordered marks. After Phase 4 normalization this
    returns False and the v1-exactness tests skip (reproduce from git history)."""
    import unicodedata

    for path in sorted((TEST_SET_DIR / "expected").rglob("*")):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            if not unicodedata.is_normalized("NFC", text):
                return True
    return False


PRE_NFC_TREE = tree_is_pre_nfc()
requires_pre_nfc_tree = pytest.mark.skipif(
    not PRE_NFC_TREE,
    reason="v1 metrics golden requires the pre-NFC test tree (checkout from git history)",
)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def golden() -> dict:
    return json.loads(GOLDEN_V1_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def golden_v2() -> dict:
    return json.loads(GOLDEN_V2_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def vocabulary() -> MajorityVocabulary:
    """v1-recipe vocabulary with the legacy parser (v1 golden reproduction)."""
    return MajorityVocabulary.from_paths([VOCABULARY_DIR], legacy=True)


@pytest.fixture(scope="session")
def stats(vocabulary: MajorityVocabulary) -> Stats:
    return Stats(basepath=TEST_SET_DIR / "expected", vocabulary=vocabulary, legacy=True)


@pytest.fixture(scope="session")
def vocabulary_v2() -> MajorityVocabulary:
    return MajorityVocabulary.from_paths([VOCABULARY_DIR])


@pytest.fixture(scope="session")
def stats_v2(vocabulary_v2: MajorityVocabulary) -> Stats:
    return Stats(basepath=TEST_SET_DIR / "expected", vocabulary=vocabulary_v2)
