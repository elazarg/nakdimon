from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort

from nakdimon import dataset, hebrew, utils
from nakdimon.config import MAIN_MODEL


@lru_cache(maxsize=4)
def load_cached_model(path: Path | str) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def merge_unconditional(texts, tnss, nss, dss, sss):
    res = []
    for ts, tns, ns, ds, ss in zip(texts, tnss, nss, dss, sss):
        sentence = []
        for t, tn, n, d, s in zip(ts, tns, ns, ds, ss):
            if tn == 0:
                break
            sentence.append(t)
            sentence.append(dataset.dagesh_table.indices_char[d] if hebrew.can_dagesh(t) else "﻿")
            sentence.append(dataset.sin_table.indices_char[s] if hebrew.can_sin(t) else "﻿")
            sentence.append(dataset.niqqud_table.indices_char[n] if hebrew.can_niqqud(t) else "﻿")
        res.append("".join(sentence))
    return res


def predict(text: str, model: ort.InferenceSession | Path | str = MAIN_MODEL, maxlen: int = 10000) -> str:
    if isinstance(model, (Path, str)):
        model = load_cached_model(model)

    data = dataset.Data.from_text(hebrew.iterate_dotted_text(text), maxlen)
    inputs = {model.get_inputs()[0].name: data.normalized.astype(np.int32)}
    n_out, d_out, s_out = model.run(None, inputs)

    actual_niqqud = dataset.from_categorical(n_out)
    actual_dagesh = dataset.from_categorical(d_out)
    actual_sin = dataset.from_categorical(s_out)

    actual = merge_unconditional(data.text, data.normalized, actual_niqqud, actual_dagesh, actual_sin)
    return " ".join(actual).replace("﻿", "").replace("  ", " ").replace(hebrew.RAFE, "")


def main(input_path: str = "-", output_path: str = "-") -> None:
    with utils.smart_open(input_path, "r", encoding="utf-8") as f:
        text = hebrew.remove_niqqud(f.read())
    text = predict(text)
    with utils.smart_open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
