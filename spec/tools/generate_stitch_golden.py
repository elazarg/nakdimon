"""Golden vectors for center-stitched windowed inference (v2 model cards).

Uses a deterministic FAKE session so the stitching logic is pinned without any
model file: the fake's argmax targets depend on (input id, batch row, position),
so overlapping windows disagree and the golden exposes exactly which window owns
each position. The JS runtime must implement the same fake in its test and
reproduce these outputs byte-for-byte.

Fake spec (both languages):
    n_id(b, t) = (input[b,t] + b + t)     % 15   # v2 niqqud classes
    d_id(b, t) = (input[b,t] + b + 2*t)   % 3
    s_id(b, t) = (input[b,t] + 2*b + t)   % 4
    logits are one-hot at the target id.

Run from the repo root:  uv run python spec/tools/generate_stitch_golden.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages" / "nakdimon" / "src"))

from nakdimon.runtime import Diacritizer, ModelCard  # noqa: E402

CARD = ModelCard(inventory="v2", window=64, pad_to_window=False, overlap=16)


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


def sample_text(n_words: int) -> str:
    alphabet = "אבגדהוזחטיכלמנסעפצקרשת"
    words = []
    for w in range(n_words):
        length = 3 + (w % 5)
        words.append("".join(alphabet[(w * 7 + i) % len(alphabet)] for i in range(length)))
    return " ".join(words)


def main() -> None:
    diacritizer = Diacritizer.from_session(FakeSession(), CARD)
    cases = []
    for n_words in [5, 14, 40]:  # single window / two windows / many windows
        text = sample_text(n_words)
        cases.append({"text": text, "output": diacritizer.diacritize(text)})

    golden = {
        "comment": "center-stitch windowed inference with the deterministic fake "
                   "session described in spec/tools/generate_stitch_golden.py; "
                   "pins window starts, ownership midpoint cuts, and NFC emission",
        "card": {"inventory": CARD.inventory, "window": CARD.window,
                 "pad_to_window": CARD.pad_to_window, "overlap": CARD.overlap},
        "cases": cases,
    }
    dest = REPO_ROOT / "spec" / "golden" / "stitch.json"
    dest.write_text(json.dumps(golden, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    print(f"wrote {dest}")
    for c in cases:
        print(len(c["text"]), "->", c["output"][:40], "...")


if __name__ == "__main__":
    main()
