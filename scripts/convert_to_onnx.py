"""One-off: convert a trained Keras .h5 / .keras model to .onnx for runtime use.

Run after `nakdimon train`. The resulting .onnx is what gets shipped in the wheel
(see [tool.setuptools.package-data] in pyproject.toml).

    python scripts/convert_to_onnx.py path/to/model.h5 nakdimon/Nakdimon.onnx
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _load_model(path: Path):
    """Load a Keras model, trying modern Keras 3 first and falling back to the
    tf-keras (Keras 2) shim for legacy models saved before the Keras 3 migration."""
    import tensorflow as tf

    try:
        return tf.keras.models.load_model(path, custom_objects={"loss": None}, compile=False)
    except (TypeError, ValueError) as e_modern:
        try:
            os.environ["TF_USE_LEGACY_KERAS"] = "1"
            import tf_keras  # type: ignore

            return tf_keras.models.load_model(path, custom_objects={"loss": None}, compile=False)
        except Exception as e_legacy:  # pragma: no cover - diagnostic
            raise RuntimeError(
                f"Failed to load {path} as either Keras 3 ({e_modern}) or "
                f"Keras 2 / tf-keras ({e_legacy})"
            ) from e_legacy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="input .h5 or .keras model")
    parser.add_argument("onnx", type=Path, help="output .onnx path")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    import tensorflow as tf
    import tf2onnx

    tf.config.set_visible_devices([], "GPU")

    model = _load_model(args.model)
    input_signature = [tf.TensorSpec([None, None], tf.int32, name="input")]

    onnx_model, _ = tf2onnx.convert.from_keras(
        model, input_signature=input_signature, opset=args.opset, output_path=str(args.onnx)
    )
    print(f"wrote {args.onnx} ({args.onnx.stat().st_size / 1e6:.1f} MB)")
    print("outputs:", [o.name for o in onnx_model.graph.output])
    return 0


if __name__ == "__main__":
    sys.exit(main())
