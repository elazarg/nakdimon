"""End-to-end smoke test for the training pipeline.

Verifies that under the modern stack (TensorFlow 2.21 / Keras 3 / wandb 0.27)
the full training flow works: build → compile → fit → save → convert-to-ONNX →
predict via onnxruntime.

Requires the [train] extra; will skip cleanly if TF is unavailable.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("WANDB_MODE", "disabled")

REPO_ROOT = Path(__file__).resolve().parents[1]

tf = pytest.importorskip("tensorflow", reason="requires [train] extra")
pytest.importorskip("tf2onnx", reason="requires [train] extra")
pytest.importorskip("wandb", reason="requires [train] extra")

# Disable GPU — smoke tests run on CPU only.
tf.config.set_visible_devices([], "GPU")


class SmokeParams:
    """Minimal training params: tiny corpus, 1 epoch per phase, batch size 8."""

    name = "Smoke"
    batch_size = 8
    units = 16  # 25x smaller than production (400) → seconds, not minutes
    validation_rate = 0
    subtraining_rate = {"premodern": 1, "modern": 1}

    # Reuse the small validation corpus for all three phases.
    corpus = {
        "premodern": ("hebrew_diacritized/validation",),
        "automatic": ("hebrew_diacritized/validation",),
        "modern": ("hebrew_diacritized/validation",),
    }

    def loss(self):
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=0)

    def epoch_params(self, data):
        # Exercise BOTH schedulers used in production:
        # CircularLearningRate (per-batch optimizer.learning_rate.assign) for premodern,
        # and the stock LearningRateScheduler for the other two phases.
        from nakdimon import schedulers

        yield (
            "premodern",
            1,
            schedulers.CircularLearningRate(3e-3, 8e-3, 1e-4, data["premodern"][0], self.batch_size),
        )
        for stage in ("automatic", "modern"):
            yield (
                stage,
                1,
                tf.keras.callbacks.LearningRateScheduler(lambda _epoch, _lr: 1e-3),
            )

    def deterministic(self):
        import numpy as np

        np.random.seed(0)
        tf.random.set_seed(0)

    def build_model(self):
        from nakdimon.dataset import DAGESH_SIZE, LETTERS_SIZE, NIQQUD_SIZE, SIN_SIZE
        from tensorflow.keras import layers

        inp = tf.keras.Input(shape=(None,), batch_size=None)
        embed = layers.Embedding(LETTERS_SIZE, self.units, mask_zero=True)(inp)
        layer = layers.Bidirectional(
            layers.LSTM(self.units, return_sequences=True, dropout=0.1), merge_mode="sum"
        )(embed)
        layer = layers.Dense(self.units)(layer)
        outputs = [
            layers.Dense(NIQQUD_SIZE, name="N")(layer),
            layers.Dense(DAGESH_SIZE, name="D")(layer),
            layers.Dense(SIN_SIZE, name="S")(layer),
        ]
        return tf.keras.Model(inputs=inp, outputs=outputs)

    def initialize_weights(self, model):
        return model


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory) -> Path:
    """Train a tiny model end-to-end via the real train.train() function."""
    from nakdimon import train as train_module

    # Run from repo root so the relative corpus paths in SmokeParams resolve.
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        model = train_module.train(SmokeParams(), group="smoke", wandb_enabled=False)
    finally:
        os.chdir(cwd)

    out = tmp_path_factory.mktemp("smoke") / "smoke.keras"
    model.save(str(out))
    assert out.exists() and out.stat().st_size > 0
    return out


def test_train_produces_model(trained_model: Path) -> None:
    """`train.train()` runs without error, returns a savable model."""
    assert trained_model.stat().st_size > 1000


@pytest.fixture(scope="module")
def converted_onnx(trained_model: Path, tmp_path_factory) -> Path:
    onnx_path = tmp_path_factory.mktemp("onnx") / "smoke.onnx"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "convert_to_onnx.py"),
            str(trained_model),
            str(onnx_path),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"},
    )
    assert result.returncode == 0, f"convert failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert onnx_path.exists() and onnx_path.stat().st_size > 0
    return onnx_path


def test_convert_to_onnx(converted_onnx: Path) -> None:
    """scripts/convert_to_onnx.py converts the freshly-trained model to .onnx."""
    assert converted_onnx.exists()


def test_predict_via_onnx_runtime(converted_onnx: Path) -> None:
    """The converted ONNX model is loadable + runnable via the production predict path."""
    import onnxruntime as ort
    from nakdimon import predict

    sess = ort.InferenceSession(str(converted_onnx), providers=["CPUExecutionProvider"])
    out = predict.predict("שלום עולם", model=sess)
    assert isinstance(out, str)
    assert len(out) > 0
