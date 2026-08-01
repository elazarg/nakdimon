"""Training loop for Nakdimon v2: AdamW, warmup+cosine, temperature-weighted
corpus mixture with a final gold-modern stage. Replaces v1's hand-tuned
per-phase learning-rate lists.

    python -m nakdimon_train.train --out models/checkpoints/v2.pt
    python -m nakdimon_train.train --smoke --out /tmp/smoke.pt
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import torch

from nakdimon_train.data import corpus
from nakdimon_train.model import ModelConfig, NakdimonV2, masked_cross_entropy, n_parameters


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 30000
    batch_size: int = 64
    lr: float = 3e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    mixture_temperature: float = 0.5   # group prob ∝ size**T
    modern_final_frac: float = 0.15    # last fraction of steps: gold modern only
    window: int = 1024
    seed: int = 2
    log_every: int = 100
    checkpoint_every: int = 1000       # periodic overwrite-save (0 = only at end)
    eval_every: int = 2000             # periodic validation accuracy (0 = only at end)
    corpus_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(corpus.CORPUS_GROUPS)
    )
    validation_path: str = corpus.VALIDATION_PATH
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


SMOKE = TrainConfig(
    steps=300,
    batch_size=32,
    warmup_steps=30,
    window=128,
    log_every=50,
    corpus_groups={"modern": ("hebrew_diacritized/validation",)},
    validation_path="hebrew_diacritized/validation/modern",
)

SMOKE_MODEL = ModelConfig(d_model=64, n_layers=2, n_heads=2, conv_blocks=1, max_context=128)


def _lr_lambda(cfg: TrainConfig):
    def f(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return f


class Mixture:
    """Temperature-weighted sampling across corpus groups; switches to the
    'modern' group only for the final fraction of training."""

    def __init__(self, cfg: TrainConfig, inventory) -> None:
        self.rng = np.random.default_rng(cfg.seed)
        self.groups = {
            name: corpus.encode_corpus(paths, cfg.window, inventory)
            for name, paths in cfg.corpus_groups.items()
        }
        sizes = np.array([len(g["letters"]) for g in self.groups.values()], dtype=np.float64)
        weights = sizes**cfg.mixture_temperature
        self.probs = weights / weights.sum()
        self.names = list(self.groups)
        self.final_only = "modern" if "modern" in self.groups else self.names[-1]
        for name, g in self.groups.items():
            logging.info(f"group {name}: {len(g['letters'])} windows")

    def batch(self, batch_size: int, *, final_stage: bool) -> dict[str, np.ndarray]:
        name = self.final_only if final_stage else self.rng.choice(self.names, p=self.probs)
        g = self.groups[name]
        idx = self.rng.integers(0, len(g["letters"]), size=batch_size)
        return {key: value[idx] for key, value in g.items()}


@torch.no_grad()
def masked_accuracy(model: NakdimonV2, data: dict[str, np.ndarray], device: str, batch_size: int = 64) -> dict[str, float]:
    model.eval()
    correct = {"N": 0, "D": 0, "S": 0}
    total = {"N": 0, "D": 0, "S": 0}
    for i in range(0, len(data["letters"]), batch_size):
        ids = torch.from_numpy(data["letters"][i : i + batch_size]).long().to(device)
        outs = dict(zip("NDS", model(ids), strict=True))
        targets = {"N": data["niqqud"], "D": data["dagesh"], "S": data["sin"]}
        for key, out in outs.items():
            y = torch.from_numpy(targets[key][i : i + batch_size]).long().to(device)
            mask = y != 0
            correct[key] += int((out.argmax(-1)[mask] == y[mask]).sum())
            total[key] += int(mask.sum())
    model.train()
    return {key: correct[key] / max(1, total[key]) for key in correct}


def train(
    cfg: TrainConfig,
    model_cfg: ModelConfig | None = None,
    init_backbone: str | None = None,
    out_path: str | None = None,
) -> NakdimonV2:
    torch.manual_seed(cfg.seed)
    inventory = corpus.load_inventory("v2")
    mixture = Mixture(cfg, inventory)
    validation = corpus.encode_corpus((cfg.validation_path,), cfg.window, inventory)
    # Periodic eval runs on a fixed subset (cheap); the final eval uses the full set.
    validation_small = {key: value[:512] for key, value in validation.items()}

    model = NakdimonV2(model_cfg)
    if init_backbone:
        state = torch.load(init_backbone, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state["backbone"], strict=False)
        logging.info(f"initialized backbone from {init_backbone} (missing {len(missing)}, unexpected {len(unexpected)})")
    model = model.to(cfg.device)
    logging.info(f"model: {n_parameters(model):,} parameters on {cfg.device}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda(cfg))

    model.train()
    final_start = int(cfg.steps * (1 - cfg.modern_final_frac))
    t0 = time.time()
    for step in range(cfg.steps):
        batch = mixture.batch(cfg.batch_size, final_stage=step >= final_start)
        ids = torch.from_numpy(batch["letters"]).long().to(cfg.device)
        n_logits, d_logits, s_logits = model(ids)
        loss = (
            masked_cross_entropy(n_logits, torch.from_numpy(batch["niqqud"]).long().to(cfg.device))
            + masked_cross_entropy(d_logits, torch.from_numpy(batch["dagesh"]).long().to(cfg.device))
            + masked_cross_entropy(s_logits, torch.from_numpy(batch["sin"]).long().to(cfg.device))
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            logging.info(
                f"step {step}/{cfg.steps} loss {loss.item():.4f} "
                f"lr {scheduler.get_last_lr()[0]:.2e} ({time.time() - t0:.0f}s)"
            )
        if out_path and cfg.checkpoint_every and step and step % cfg.checkpoint_every == 0:
            save_checkpoint(model, cfg, out_path)
        if cfg.eval_every and step and step % cfg.eval_every == 0:
            acc = masked_accuracy(model, validation_small, cfg.device)
            logging.info(f"step {step} validation(sub): N {acc['N']:.4f} D {acc['D']:.4f} S {acc['S']:.4f}")

    acc = masked_accuracy(model, validation, cfg.device)
    logging.info(f"validation masked accuracy: N {acc['N']:.4f} D {acc['D']:.4f} S {acc['S']:.4f}")
    return model


def save_checkpoint(model: NakdimonV2, cfg: TrainConfig, path: str) -> None:
    torch.save(
        {"model": model.state_dict(), "config": asdict(model.cfg), "train_config": asdict(cfg)},
        path,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="checkpoint output path (.pt)")
    parser.add_argument("--smoke", action="store_true", help="tiny model + tiny corpus, minutes on CPU")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--init", default=None, help="optional pretrained backbone checkpoint (pretrain.py)")
    args = parser.parse_args()

    cfg = SMOKE if args.smoke else TrainConfig()
    for name in ("steps", "window", "batch_size", "warmup_steps", "checkpoint_every", "eval_every"):
        value = getattr(args, name)
        if value is not None:
            cfg = replace(cfg, **{name: value})
    model_cfg = SMOKE_MODEL if args.smoke else None

    model = train(cfg, model_cfg, init_backbone=args.init, out_path=args.out)
    save_checkpoint(model, cfg, args.out)
    logging.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
