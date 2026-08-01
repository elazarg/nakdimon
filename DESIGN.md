# Nakdimon v2 — Design

Status: accepted, scaffolding in progress. This document is the reference for the
v2 rebuild; the migration checklist at the bottom tracks what exists vs. planned.

## 1. The contract (what v2 preserves)

1. **IO**: undotted Hebrew text in → the same text with niqqud/dagesh/sin layered on,
   `remove_niqqud(output)` round-trips to the (cleaned) input modulo whitespace collapsing.
2. **IO shape**: per-character classification with three independent heads —
   niqqud, dagesh, sin — decoded by argmax. Deterministic, O(n), alignment by
   construction. Deliberately **not** a seq2seq/LLM formulation: generative models
   can drop/insert letters and, empirically, produce nikud that is "too good" —
   normative, ktiv-haser-flavored conventions rather than this project's register
   (see §6, *output convention*).
3. **Standalone**: no dictionary, no morphology, no network at inference time.
4. **JS-executable**: the same model artifact runs fully client-side in a browser.

Everything else — frameworks, formats, training recipe, repo layout — is replaced.

## 2. System shape

One compiled model artifact (ONNX), thin runtimes, and a machine-readable spec
as the single source of truth for the text layer:

```
spec/
  hebrew.json          # normalization, inventories, decomposition rules, mark order
  golden/*.json        # byte-exact vectors every runtime must reproduce
  tools/generate_spec.py  # generated the spec FROM the v1 code (the oracle)
packages/
  nakdimon/            # (py)  runtime: onnxruntime + a few hundred lines; `diacritize` CLI
  nakdimon-js/         # (ts)  same text layer + onnxruntime-web wrapper; npm package
  nakdimon-train/      # (py)  PyTorch: model, export, data pipeline, ablations
models/                # released .onnx (fp16 + int8) + sidecar model-card JSON
web/                   # static demo built on nakdimon-js
```

- **ONNX is the only model format.** Python (`onnxruntime`), browser and Node
  (`onnxruntime-web`, WASM + WebGPU) consume the identical file. The `.keras`
  runtime path, TF.js shards (`model_js/`), and conversion shims are gone.
- **PyTorch replaces TensorFlow** for research. `torch.onnx.export` with dynamic
  axes replaces the tf2onnx script and the tf-keras compatibility shims; the
  Python-version ceiling disappears.
- **Runtimes are ports of a spec, not of each other.** CI runs the same golden
  vectors against Python and TS; drift is a test failure, not a code comment
  ("make sure to be consistent with JS").

## 3. The spec and what v1 actually does (findings)

`spec/hebrew.json` was generated from the v1 code with every claim asserted
against it. Non-obvious facts now recorded explicitly:

- **Final letters are first-class vocabulary.** `normalize()` passes ךםןףץ
  through unchanged (they are in `VALID_LETTERS`); the `ENDINGS_TO_REGULAR`
  fold was dead code. The model has always seen finals as distinct ids.
- **The niqqud encode/decode asymmetry.** v1's `CharacterTable` contains PATAH
  twice (decode indices 9 and 15). Dict-overwrite made *encoding* always produce
  15, so the trained model predicts 15 and index 9 is dead. Any runtime loading
  a v1-inventory model must encode patah as 15. Captured as
  `inventories.v1.niqqud.encode_overrides`.
- **U+05BA (holam haser for vav)** sits in the v1 inventory but essentially never
  occurs in data. v2 drops the class and folds it to holam during data prep.
- **v1 mark order is not NFC; v2 is NFC-native.** v1 emitted
  dagesh → sin-dot → niqqud, but Unicode canonical order is niqqud (ccc 10–20)
  < dagesh (21) < shin/sin dots (24/25), and NFC-ordered text *crashes* v1's
  strict parser. Rather than enshrine the wart, v2 fixes it globally: **emit NFC
  canonical order** (a fixpoint — Hebrew presentation forms are composition
  exclusions, so NFC is a pure reorder and any standards-compliant pipeline
  leaves our output untouched), **parse marks in any order** (last-wins per
  channel), and compare text in metrics via decomposed form, never raw strings.
  The model is unaffected (marks are labels, not input). Legacy corpora and
  test sets stay in v1 order until v1 tooling retires — the v2 parser reads
  both — and are NFC-normalized in Phase 4.
- **Digit rule.** v1 used `str.isdigit()` (which matches some non-Nd chars like
  '²'); v2 restricts to Unicode category Nd. The only intentional normalization
  divergence; listed in the spec.

## 4. Inventories and qamats qatan

Models declare their inventory in a sidecar model card; the runtime assumes nothing.

- **v1** (16 niqqud classes incl. the dead duplicate): what the shipped
  `Nakdimon.onnx` uses. Kept forever for compatibility.
- **v2** (15 classes): duplicate removed, U+05BA dropped, **qamats qatan (U+05C7)
  added as a first-class label**.

Qamats qatan is in because it is genuinely useful (TTS, reading aids) and cheap at
the model level — but the *data* is the hard part: existing corpora type it as
plain qamats. The plan:

1. Relabel the modern corpora through Dicta's nakdan with `keepqq: true`
   (the v1 fetcher already passes `keepqq: false` — the switch literally exists),
   projecting only the qamats→qamats-qatan substitutions back onto the gold text
   so nothing else changes. Tokens where the teacher disagrees with gold on
   anything *other* than qq are left untouched.
2. Spot-check a sample (qq is rare enough that manual review of the full
   substitution list is feasible).
3. Until a corpus snapshot passes that pipeline, training folds U+05C7 → qamats
   and the class is present-but-untrained. Evaluation always reports the folded
   metrics too, so v2 numbers stay comparable with v1 and published results.

## 5. Model

Same interface (char ids in; N/D/S logits out), new body:

- **Conv stem + pre-norm Transformer encoder with RoPE.** Embedding (d≈320) →
  two depthwise-separable conv blocks (kernel 5; local n-gram bias) → 6 encoder
  layers (4 heads, GELU, pre-norm, RoPE) → three linear heads. ≈6–7M params vs.
  v1's ≈10M BiLSTM stack.
- Why not keep the BiLSTM: poor int8 quantization, weak ORT-Web kernels, no
  parallel scan, hard context ceiling. Why not SSM/Mamba: bidirectionality needs
  two passes and ORT-Web op coverage is the one risk this project cannot take.
- **Context 1024 chars** (v1: 80), free length axis in the export. Register cues,
  construct chains, and qamats-qatan disambiguation live beyond 80 chars.
- Ship fp16 (WebGPU) and dynamic-int8 (CPU/WASM) variants; target ≤8 MB download
  (v1 TF.js shards: ~24 MB). QAT only if int8 costs > 0.1pp DEC.

## 6. Training and data

The v1 ablations showed corpus mixture beats architecture; v2 invests accordingly.

1. **Self-supervised pretraining**: masked-char denoising on large undotted
   modern Hebrew (Wikipedia/OSCAR scale). Free data for a char encoder; expected
   to be the biggest OOV lever. (v1's `pretrain.py` gestured at this.)
2. **Silver distillation at scale**: label a large modern corpus with the best
   available teacher, train the student on silver, fine-tune on gold modern.
   ("Dictionary-free" is an inference property; the thesis survives.)
3. **Boring, robust optimization**: AdamW, warmup + cosine, bf16,
   temperature-weighted mixture sampling over corpora with a final gold-modern
   stage — replacing the hand-tuned per-phase LR lists and `CircularLearningRate`.

**Output convention — the "full script" constraint.** The corpus register is
modern **ktiv male** spelling with diacritics layered onto the letters *as
written*. Normative nikud implies different spelling (ktiv haser: מילים→מִלִּים),
meteg, fuller marking. LLMs and dictionary-driven tools drift toward that
register — output that is "too correct" for this dataset and breaks the
alignment contract. Therefore every teacher goes through a **convention
projector** (`nakdimon_train.data.conventions`):

- *Alignment projection*: teacher output is aligned to the exact input letter
  sequence; a token whose letters differ from the input (inserted/removed
  matres lectionis, respelled words) is **rejected**, not repaired.
- *Convention normalization*: the mark-level rewrites v1 applied ad-hoc to Dicta
  output (meteg removal, kubuts+vav→shuruk, holam+vav rewrites, double-dagesh
  collapse, …) become a single named, tested transform shared by the silver
  pipeline, the qq relabeler, and evaluation of external systems.

## 7. Inference

- Sliding **1024-char windows, ~128 overlap, center-stitching**: each char takes
  its prediction from the window where it is most central. Kills the v1
  boundary-artifact class of errors. Parameters live in the model card.
- **v1-compat mode**: the shipped v1 model is served exactly as v1 did
  (single 10000-char window, zero-padded to full width — the bidirectional pass
  reads padding, so padding is behaviorally load-bearing). `predict_v1.json`
  golden-pins this byte-for-byte, with outputs stored in NFC: v2 runtimes emit
  NFC natively, so equality against raw v1 output holds only after NFC on both
  sides.
- Heads are consulted only where `can_niqqud/can_dagesh/can_sin` allow.
- No server. Python CLI + library; browser fully client-side. A FastAPI
  one-liner can live in `examples/` for whoever wants HTTP.

## 8. Evaluation and CI gates

- DEC/CHA/WOR/VOC (+OOV variants) stay canonical, ported into `nakdimon-train`.
  With v2 inventory, metrics are reported both raw and qq-folded (§4).
- CI gates: (a) **golden parity** — Python and TS byte-identical on
  `spec/golden/`; (b) **model regression** — pinned metric floors on
  `tests/validation` for each released model; (c) **spec sync** — packaged spec
  copies equal `spec/hebrew.json`.
- External systems (Dicta/Snopi/Morfix fetchers) become an optional research
  extra used only to reproduce the paper tables; leakage guards
  (MajAllNoDicta on the dicta test set) stay encoded in the eval driver.

## 9. Migration plan

Phase 0 — DONE: spec + golden vectors generated from v1 as oracle.
Phase 1 — DONE: `packages/nakdimon` (py) and `packages/nakdimon-js` (ts) pass
goldens + byte-exact v1-model e2e; bundled model + explicit v1 card.
Phase 2 — DONE: metrics ported (0.0 deviation vs the v1 oracle in legacy mode;
`metrics_v2.json` pins the going-forward defaults), eval driver with leakage
guard, corpus reader on the v2 text layer.
Phase 3 — DONE except production training: full stack works end-to-end (smoke
train → dynamic int32 ONNX → v2 model card → center-stitched inference, ONNX↔
torch argmax parity 1.0); pretraining + silver pipeline + qq relabeler coded.
Remaining: train the real v2 model on a GPU (pretrain → silver → gold), then
int8/fp16 variants and a metric-floor regression gate for it.
Phase 4 — DONE except publishing: v1 package/TF stack/`model_js/`/`other/`
deleted (branch `v1-archive` preserves the tree), uv workspace root,
`web/` rebuilt on `nakdimon-js`, corpora + tests NFC-normalized with per-file
verified parse-invariance (13 files with biblical-style multi-vowel letters
kept legacy order — see §3 note below). Remaining: PyPI 2.0 + npm publish.

**Metrics parser note.** v1's positional grammar mis-parses a double-dagesh
emission glitch present in some archived system outputs; the port reproduces it
(`tokens.decompose_legacy`, `legacy=True`) solely to pin `metrics_v1.json`.
Default metrics use the order-tolerant parser and sorted iteration
(`metrics_v2.json`) — values are NFC-invariant, verified by re-running after
the corpus normalization. **Multi-vowel letters** (ketiv/qere composites like
ירושָׁלִַם) are the one case where NFC changes a last-wins parse; such files are
left in legacy order rather than silently altered.

## 10. Open items

- Teacher licensing for silver data at scale (Dicta ToS for bulk labeling).
- Qamats qatan relabel QA protocol and acceptance threshold (§4).
- Whether the spec eventually *generates* the table modules instead of testing
  them (start with testing; generate only if drift actually happens).
- npm package name (`nakdimon` vs scoped) — decide before first publish.
