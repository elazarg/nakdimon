"""Golden-vector tests: spec/golden/*.json pins the exact behavior of the text layer
and encoder, independent of any model. See spec/hebrew.json for the rules these vectors
exercise (normalization table, decomposition order-tolerance, the v1 patah encode
override, ...).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nakdimon import hebrew
from nakdimon.encode import Inventory
from nakdimon.spec import load_spec

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_DIR = REPO_ROOT / "spec" / "golden"


def _load_golden(name: str):
    return json.loads((GOLDEN_DIR / name).read_text(encoding="utf-8"))


INVENTORY_V1 = Inventory(load_spec()["inventories"]["v1"])


@pytest.mark.parametrize("case", _load_golden("normalize.json"))
def test_normalize_text(case: dict) -> None:
    assert hebrew.normalize_text(case["input"]) == case["normalized"]


@pytest.mark.parametrize("case", _load_golden("remove_niqqud.json"))
def test_remove_niqqud(case: dict) -> None:
    assert hebrew.remove_niqqud(case["input"]) == case["output"]
    assert hebrew.remove_niqqud(case["input_nfc"]) == case["output"]


@pytest.mark.parametrize("case", _load_golden("decompose.json"))
def test_decompose(case: dict) -> None:
    items = hebrew.decompose(case["dotted"])

    assert [list(item) for item in items] == case["items"]
    assert hebrew.items_to_text(items) == case["reconstructed"]
    # Order tolerance: NFC-ordered input must decompose to the very same items.
    assert hebrew.decompose(case["dotted_nfc"]) == items
    # Emission is an NFC fixpoint: re-parsing our own output is the identity.
    assert hebrew.items_to_text(hebrew.decompose(case["reconstructed"])) == case["reconstructed"]

    encoded = INVENTORY_V1.encode_items(items)
    assert encoded["niqqud"] == case["niqqud_ids"]
    assert encoded["dagesh"] == case["dagesh_ids"]
    assert encoded["sin"] == case["sin_ids"]


@pytest.mark.parametrize("case", _load_golden("encode_v1.json"))
def test_encode_letters_v1(case: dict) -> None:
    assert hebrew.normalize_text(case["text"]) == case["normalized_flat"]
    assert INVENTORY_V1.encode_letters(case["normalized_flat"]) == case["ids"]
