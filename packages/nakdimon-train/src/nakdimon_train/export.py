"""Export a trained NakdimonV2 to ONNX plus its sidecar model card.

The model card is the contract between a model file and the runtimes: it names
the inventory (v1/v2 — see spec/hebrew.json) and the windowing parameters, so
runtimes never assume. The v1 shipped model has no card and runtimes default to
v1 semantics (window 10000, pad-to-window).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from nakdimon_train.model import ModelConfig, NakdimonV2


def export_onnx(model: NakdimonV2, path: Path, opset: int = 18) -> None:
    model.eval()
    # int32 input is the runtime contract (v1 model compatibility; JS feeds int32).
    dummy = torch.ones(1, 64, dtype=torch.int32)
    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        input_names=["input"],
        output_names=["N", "D", "S"],
        dynamic_shapes={"ids": {0: torch.export.Dim.DYNAMIC, 1: torch.export.Dim.DYNAMIC}},
        opset_version=opset,
    )


def write_model_card(
    onnx_path: Path,
    cfg: ModelConfig,
    *,
    inventory: str = "v2",
    window: int = 1024,
    overlap: int = 128,
    train_run: str | None = None,
) -> Path:
    card = {
        "inventory": inventory,
        "window": window,
        "overlap": overlap,
        "pad_to_window": False,
        "stitching": "center",
        "config": cfg.__dict__,
        "train_run": train_run,
        "sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
    }
    card_path = onnx_path.with_suffix(onnx_path.suffix + ".json")
    card_path.write_text(json.dumps(card, indent=1) + "\n", encoding="utf8")
    return card_path


def main(checkpoint: str, out: str) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = ModelConfig(**state["config"]) if "config" in state else ModelConfig()
    model = NakdimonV2(cfg)
    model.load_state_dict(state["model"] if "model" in state else state)
    out_path = Path(out)
    export_onnx(model, out_path)
    write_model_card(out_path, cfg, train_run=state.get("run_name"))
    print(f"wrote {out_path} + model card")


if __name__ == "__main__":
    import sys

    main(sys.argv[1], sys.argv[2])
