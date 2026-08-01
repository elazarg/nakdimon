# Nakdimon: a simple Hebrew diacritizer

Repository for [Restoring Hebrew Diacritics Without a Dictionary](https://arxiv.org/abs/2105.05209)
by Elazar Gershuni and Yuval Pinter — rebuilt as **v2**: one ONNX model artifact,
thin spec-driven runtimes for Python and JavaScript, and a PyTorch research stack.
See `DESIGN.md` for the architecture and migration status.

Demo: https://nakdimon.org/

## Use it (Python)

```
$ pip install nakdimon
$ diacritize input.txt -o output.txt      # or: echo "שלום עולם" | diacritize -
```

```python
import nakdimon
nakdimon.diacritize("שלום עולם")            # bundled model
nakdimon.diacritize(text, "my_model.onnx")  # any model + optional .onnx.json card
```

## Use it (JavaScript)

`packages/nakdimon-js` — same text layer and runtime, byte-identical output
(enforced by shared golden vectors). Works in the browser via onnxruntime-web
and in Node via onnxruntime-node:

```js
import { createDiacritizer } from "nakdimon";
const d = await createDiacritizer(ort, "Nakdimon.onnx");
await d.diacritize("שלום עולם");
```

The static demo in `web/` is exactly this (`web/build.sh`, then serve `web/`).

## Repository layout

```
spec/               source of truth: hebrew.json (normalization, inventories,
                    NFC mark-order policy) + golden vectors + generator tools
packages/
  nakdimon/         Python runtime (onnxruntime only) + bundled model
  nakdimon-js/      TypeScript runtime (npm package)
  nakdimon-train/   PyTorch: model, training, pretraining, silver data, metrics
models/             released model artifacts + sidecar model cards
hebrew_diacritized/ training corpus        tests/  test sets (expected/ = gold)
web/                client-side demo
```

Output convention: full-script (ktiv male) diacritics layered on the input
letters as written, emitted in NFC canonical mark order; `remove_niqqud(output)`
round-trips the input. Runtimes parse marks in any order.

## Train / evaluate (research)

```
$ uv sync --project packages/nakdimon-train        # CPU torch by default
$ uv run --project packages/nakdimon-train python -P -m nakdimon_train.train \
      --out models/checkpoints/v2.pt               # --smoke for a tiny run
$ uv run --project packages/nakdimon-train python -P -c "
from nakdimon_train.export import main; main('models/checkpoints/v2.pt', 'models/NakdimonV2.onnx')"
$ uv run --project packages/nakdimon-train python -P -m nakdimon_train.eval \
      --test-set tests/new
```

- Training: AdamW, warmup+cosine, temperature-weighted corpus mixture with a
  final gold-modern stage (`nakdimon_train/train.py`).
- Pretraining on undotted text: `nakdimon_train/pretrain.py` (then `--init`).
- Silver data: `nakdimon_train/data/silver.py` — every teacher passes the
  convention projector (alignment-preserving; respelled tokens rejected).
- Qamats qatan relabeling: `nakdimon_train/data/qamats_qatan.py` (Dicta
  `keepqq`); the v2 label inventory includes U+05C7, folded to qamats until a
  relabeled corpus snapshot exists.
- Metrics: DEC/CHA/WOR/VOC (+OOV), pinned to the v1 implementation by
  `spec/golden/metrics_v1.json`.

Test sets: `tests/new` (paper) and `tests/dicta`; each has an `expected/` gold
folder and one folder per system. On `tests/dicta`, Nakdimon and
`MajAllWithDicta` are refused (train/test leakage guard).

## Tests

```
$ uv run --project packages/nakdimon --with pytest pytest packages/nakdimon/tests
$ cd packages/nakdimon-js && npm test
$ uv run --with pytest python -m pytest packages/nakdimon-train/tests
```

Golden parity (both runtimes byte-identical on `spec/golden/`, including real
model inference) is the compatibility contract; regenerating goldens requires
the v1 oracle from git history (pre-v2 checkout).

## Citation

```bibtex
@inproceedings{gershuni2022restoring,
  title={Restoring Hebrew Diacritics Without a Dictionary},
  author={Gershuni, Elazar and Pinter, Yuval},
  booktitle={Findings of the Association for Computational Linguistics: NAACL 2022},
  pages={1010--1018},
  year={2022}
}
```
