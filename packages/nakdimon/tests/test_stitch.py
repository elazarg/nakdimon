"""Center-stitched windowed inference against spec/golden/stitch.json.

The fake session spec lives in spec/tools/generate_stitch_golden.py; it is
deliberately re-implemented here (and in the JS test) rather than imported, so
each runtime's test stands alone against the golden."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nakdimon import hebrew
from nakdimon.runtime import Diacritizer, ModelCard

GOLDEN = json.loads(
    (Path(__file__).resolve().parents[3] / "spec" / "golden" / "stitch.json").read_text(encoding="utf-8")
)


class _FakeInput:
    name = "input"


class FakeSession:
    def get_inputs(self):
        return [_FakeInput()]

    def run(self, _outputs, feeds):
        x = feeds["input"]
        b_idx = np.arange(x.shape[0])[:, None]
        t_idx = np.arange(x.shape[1])[None, :]

        def one_hot(targets: np.ndarray, classes: int) -> np.ndarray:
            return np.eye(classes, dtype=np.float32)[targets % classes]

        return (
            one_hot(x + b_idx + t_idx, 15),
            one_hot(x + b_idx + 2 * t_idx, 3),
            one_hot(x + 2 * b_idx + t_idx, 4),
        )


@pytest.fixture(scope="module")
def diacritizer() -> Diacritizer:
    card = ModelCard(**GOLDEN["card"])
    return Diacritizer.from_session(FakeSession(), card)


@pytest.mark.parametrize("case", GOLDEN["cases"])
def test_stitch_golden(diacritizer: Diacritizer, case: dict) -> None:
    assert diacritizer.diacritize(case["text"]) == case["output"]


@pytest.mark.parametrize("case", GOLDEN["cases"])
def test_stitch_roundtrip_and_nfc(diacritizer: Diacritizer, case: dict) -> None:
    out = diacritizer.diacritize(case["text"])
    # Alignment contract: stripping marks recovers the input letters exactly.
    assert hebrew.remove_niqqud(out) == case["text"]
    # Emission is an NFC fixpoint.
    import unicodedata

    assert unicodedata.normalize("NFC", out) == out
