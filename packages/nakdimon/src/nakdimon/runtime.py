"""ONNX inference wrapper. Mirrors the v1 predict pipeline (nakdimon/predict.py +
nakdimon/dataset.py) exactly, so that byte-exact reproduction of the shipped v1
model's outputs (spec/golden/predict_v1.json) is a build invariant, not an accident.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import onnxruntime as ort

from nakdimon.encode import Inventory
from nakdimon.hebrew import RAFE, HebrewItem, can_dagesh, can_niqqud, can_sin, clean_text, decompose
from nakdimon.spec import load_spec


@dataclass(frozen=True)
class ModelCard:
    inventory: str = "v1"
    window: int = 10000
    pad_to_window: bool = True
    overlap: int = 0

    @staticmethod
    def load(model_path: str | Path) -> ModelCard:
        sidecar = Path(f"{model_path}.json")
        if not sidecar.exists():
            return ModelCard()
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        # Sidecars may carry extra provenance fields (config, sha256, train_run, ...);
        # only the windowing/inventory contract is this runtime's concern.
        known = {f.name for f in fields(ModelCard)}
        return ModelCard(**{k: v for k, v in data.items() if k in known})


def _split_by_length(items: list[HebrewItem], maxlen: int) -> Iterator[list[HebrewItem]]:
    """Chunk items into windows of at most `maxlen`, preferring to break on the last
    space seen so words are never split across chunk boundaries. Ported verbatim from
    v1's hebrew.split_by_length: a subtly different boundary rule changes chunking and
    therefore the model's output.
    """
    assert maxlen > 1
    out: list[HebrewItem] = []
    space = maxlen
    for item in items:
        if item.letter == " ":
            space = len(out)
        out.append(item)
        if len(out) == maxlen - 1:
            yield out[: space + 1]
            out = out[space + 1 :]
    if out:
        yield out


def _window_starts(n: int, window: int, overlap: int) -> list[int]:
    """Start offsets of overlapping windows covering [0, n). The last window is
    right-aligned so no position is ever closer than `overlap`/2 to a hard edge
    (except the text's own ends)."""
    if n <= window:
        return [0]
    stride = window - overlap
    starts = list(range(0, n - window, stride))
    starts.append(n - window)
    return starts


def _ownership_cuts(starts: list[int], window: int, n: int) -> list[int]:
    """Exclusive end of each window's owned region: consecutive windows meet at
    the midpoint of their overlap, so every position is predicted by the window
    in which it is most central. Integer floor midpoint — keep identical to the
    JS port."""
    cuts = [(starts[k + 1] + starts[k] + window) // 2 for k in range(len(starts) - 1)]
    cuts.append(n)
    return cuts


class Diacritizer:
    def __init__(self, model_path: str | Path, providers: list[str] | None = None) -> None:
        session = ort.InferenceSession(str(model_path), providers=providers or ["CPUExecutionProvider"])
        self._init(session, ModelCard.load(model_path))

    @classmethod
    def from_session(cls, session, card: ModelCard) -> Diacritizer:
        """Wrap an existing (possibly fake) session — the testing/embedding hook."""
        self = cls.__new__(cls)
        self._init(session, card)
        return self

    def _init(self, session, card: ModelCard) -> None:
        self.session = session
        self.card = card
        self.inventory = Inventory(load_spec()["inventories"][card.inventory])
        self._input_name = session.get_inputs()[0].name

    def _run(self, input_array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_out, d_out, s_out = self.session.run(None, {self._input_name: input_array})
        return n_out.argmax(axis=-1), d_out.argmax(axis=-1), s_out.argmax(axis=-1)

    def _emit(self, item: HebrewItem, niqqud_id: int, dagesh_id: int, sin_id: int) -> str:
        # NFC canonical emission order: niqqud, dagesh, sin-dot (spec["mark_order"]).
        parts = [item.letter]
        if can_niqqud(item.letter):
            parts.append(self.inventory.decode_niqqud(niqqud_id))
        if can_dagesh(item.letter):
            parts.append(self.inventory.decode_dagesh(dagesh_id))
        if can_sin(item.letter):
            parts.append(self.inventory.decode_sin(sin_id))
        return "".join(parts)

    def diacritize(self, text: str) -> str:
        items = decompose(clean_text(text))
        if not items:
            return ""
        if not self.card.pad_to_window:
            return self._diacritize_stitched(items)
        return self._diacritize_v1(items)

    def _diacritize_stitched(self, items: list[HebrewItem]) -> str:
        """v2 inference: overlapping windows, each position predicted by the
        window where it is most central. Windows are char-aligned (no word
        snapping) — attention context makes boundary words a non-issue, and the
        ownership region never touches a window edge."""
        card = self.card
        assert 0 <= card.overlap < card.window, "model card: need 0 <= overlap < window"
        ids = self.inventory.encode_letters("".join(item.normalized for item in items))
        n = len(ids)
        starts = _window_starts(n, card.window, card.overlap)
        # Rows are always full window width (zero-padded): keeps the input shape
        # fixed so trace-exported models with a baked sequence length work too.
        width = card.window
        input_array = np.zeros((len(starts), width), dtype=np.int32)
        for row, s in enumerate(starts):
            input_array[row, : min(width, n - s)] = ids[s : s + width]

        niqqud_ids, dagesh_ids, sin_ids = self._run(input_array)

        cuts = _ownership_cuts(starts, card.window, n)
        out: list[str] = []
        own_start = 0
        for row, s in enumerate(starts):
            for pos in range(own_start, cuts[row]):
                local = pos - s
                out.append(self._emit(items[pos], niqqud_ids[row, local], dagesh_ids[row, local], sin_ids[row, local]))
            own_start = cuts[row]
        return "".join(out).replace(RAFE, "")

    def _diacritize_v1(self, items: list[HebrewItem]) -> str:
        chunks = list(_split_by_length(items, self.card.window))
        if not chunks:
            return ""

        rows = [self.inventory.encode_letters("".join(item.normalized for item in chunk)) for chunk in chunks]
        width = self.card.window
        input_array = np.zeros((len(chunks), width), dtype=np.int32)
        for i, row in enumerate(rows):
            input_array[i, : len(row)] = row

        niqqud_ids, dagesh_ids, sin_ids = self._run(input_array)

        chunk_texts = []
        for chunk_index, chunk in enumerate(chunks):
            chunk_texts.append(
                "".join(
                    self._emit(item, niqqud_ids[chunk_index, pos], dagesh_ids[chunk_index, pos], sin_ids[chunk_index, pos])
                    for pos, item in enumerate(chunk)
                )
            )

        return " ".join(chunk_texts).replace("  ", " ").replace(RAFE, "")
