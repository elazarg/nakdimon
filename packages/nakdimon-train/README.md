# nakdimon-train

PyTorch training/export stack for Nakdimon v2. See ../../DESIGN.md for the full
design; this package is Phase 1 scaffolding.

## Present

- `model.py` — conv stem + pre-norm Transformer encoder with RoPE (~6.5M params
  at defaults), N/D/S heads, `masked_cross_entropy` mirroring v1's
  `ignore_class=0` (RAFE is a real supervised label).
- `export.py` — ONNX export (dynamic batch/seq axes, output order N, D, S) and
  the sidecar model card that tells runtimes the inventory + windowing contract.
- `data/conventions.py` — convention projector: teacher output → corpus
  register (full-script/ktiv-male), alignment-preserving by construction;
  respelled tokens are rejected, not repaired.
- `data/qamats_qatan.py` — qq relabeler: adopt *only* qamats→qamats-qatan
  substitutions from a keepqq teacher, with a reviewable substitution report.

## Not yet (DESIGN.md §9, Phases 2–3)

- Corpus reader / window sampler on top of the `nakdimon` v2 text layer.
- Metrics port (DEC/CHA/WOR/VOC + OOV, qq-folded variants) and the eval driver
  with leakage guards.
- Masked-char pretraining and the silver-distillation pipeline.
- Train loop (AdamW, warmup+cosine, bf16, temperature-weighted corpus mixture).

## Setup

Requires torch; not installed in the repo's runtime venv by design:

    uv sync --project packages/nakdimon-train
