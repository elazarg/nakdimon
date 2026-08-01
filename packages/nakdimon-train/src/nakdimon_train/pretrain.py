"""Masked-char pretraining on undotted Hebrew (DESIGN.md §6.1).

Free data for the char encoder: mask 15% of letter ids (80% -> mask token 0,
10% -> random letter, 10% -> kept) and predict the original id. The resulting
backbone initializes fine-tuning via `train.py --init`.

    python -m nakdimon_train.pretrain --corpus path/to/undotted --out models/checkpoints/backbone.pt
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from nakdimon import hebrew
from nakdimon_train.data import corpus
from nakdimon_train.model import ModelConfig, NakdimonV2

MASK_ID = 0  # shared with padding, as in v1's CharacterTable; loss covers only masked positions


@dataclass(frozen=True)
class PretrainConfig:
    steps: int = 100000
    batch_size: int = 64
    lr: float = 6e-4
    warmup_steps: int = 2000
    weight_decay: float = 0.01
    mask_rate: float = 0.15
    window: int = 1024
    seed: int = 2
    log_every: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PretrainModel(nn.Module):
    def __init__(self, model_cfg: ModelConfig | None = None):
        super().__init__()
        self.backbone = NakdimonV2(model_cfg)
        self.head_letters = nn.Linear(self.backbone.cfg.d_model, self.backbone.cfg.n_letters)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head_letters(self.backbone.features(ids))


def encode_undotted(paths: tuple[str, ...], window: int, inventory) -> np.ndarray:
    """Undotted corpus -> [N, window] letter ids (zero-padded rows)."""
    rows = []
    for path in corpus.iter_files(paths):
        text = "".join(word + " " for word in path.read_text(encoding="utf-8").split())
        normalized = hebrew.normalize_text(hebrew.remove_niqqud(hebrew.clean_text(text)))
        ids = inventory.encode_letters(normalized)
        for start in range(0, len(ids), window):
            row = np.zeros(window, dtype=np.int32)
            chunk = ids[start : start + window]
            row[: len(chunk)] = chunk
            rows.append(row)
    if not rows:
        raise ValueError(f"no corpus content under {paths}")
    return np.stack(rows)


def mask_batch(
    ids: np.ndarray, rng: np.random.Generator, mask_rate: float, n_letters: int
) -> tuple[np.ndarray, np.ndarray]:
    """BERT-style corruption; targets are 0 (=ignored) except at masked positions."""
    real = ids != 0
    chosen = (rng.random(ids.shape) < mask_rate) & real
    corrupted = ids.copy()
    roll = rng.random(ids.shape)
    corrupted[chosen & (roll < 0.8)] = MASK_ID
    random_positions = chosen & (roll >= 0.8) & (roll < 0.9)
    corrupted[random_positions] = rng.integers(1, n_letters, size=int(random_positions.sum()))
    targets = np.where(chosen, ids, 0)
    return corrupted, targets


def pretrain(cfg: PretrainConfig, paths: tuple[str, ...], model_cfg: ModelConfig | None = None) -> PretrainModel:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    inventory = corpus.load_inventory("v2")
    data = encode_undotted(paths, cfg.window, inventory)
    logging.info(f"pretraining on {len(data)} windows")

    model = PretrainModel(model_cfg).to(cfg.device)
    n_letters = model.backbone.cfg.n_letters
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    for step in range(cfg.steps):
        idx = rng.integers(0, len(data), size=cfg.batch_size)
        corrupted, targets = mask_batch(data[idx], rng, cfg.mask_rate, n_letters)
        logits = model(torch.from_numpy(corrupted).long().to(cfg.device))
        y = torch.from_numpy(targets).long().to(cfg.device)
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            logging.info(f"step {step}/{cfg.steps} mlm-loss {loss.item():.4f}")
    return model


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", nargs="+", required=True, help="undotted text dirs/files")
    parser.add_argument("--out", required=True, help="backbone checkpoint output (.pt)")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    cfg = PretrainConfig() if args.steps is None else PretrainConfig(steps=args.steps)
    model = pretrain(cfg, tuple(args.corpus))
    torch.save(
        {"backbone": model.backbone.state_dict(), "config": asdict(model.backbone.cfg)},
        args.out,
    )
    logging.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
