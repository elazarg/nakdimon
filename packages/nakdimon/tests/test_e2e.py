"""End-to-end test: the v2 runtime, driving the v1 shipped ONNX model, must reproduce
v1's predict() output byte-for-byte. This is the whole point of runtime.py mirroring
the v1 pipeline (chunking, padding, decode) exactly rather than approximately.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ort = pytest.importorskip("onnxruntime")

from nakdimon.runtime import Diacritizer  # noqa: E402 — after importorskip

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = Path(__file__).resolve().parents[1] / "src" / "nakdimon" / "data" / "Nakdimon.onnx"
GOLDEN_PATH = REPO_ROOT / "spec" / "golden" / "predict_v1.json"


@pytest.fixture(scope="module")
def diacritizer() -> Diacritizer:
    if not MODEL_PATH.exists():
        pytest.skip(f"v1 model not present at {MODEL_PATH}")
    return Diacritizer(MODEL_PATH)


def _cases() -> list[dict]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", _cases())
def test_predict_v1_byte_exact(diacritizer: Diacritizer, case: dict) -> None:
    assert diacritizer.diacritize(case["input"]) == case["output"]
