"""Nakdimon v2: Hebrew niqqud (diacritics) restoration."""

from __future__ import annotations

import os
from importlib.resources import as_file, files

from nakdimon.runtime import Diacritizer

__version__ = "2.0.0a0"

_cache: dict[str, Diacritizer] = {}


def default_model_path() -> str:
    """NAKDIMON_MODEL env var if set, else the bundled model (wheel package data)."""
    env = os.environ.get("NAKDIMON_MODEL")
    if env:
        return env
    with as_file(files("nakdimon").joinpath("data", "Nakdimon.onnx")) as p:
        return str(p)


def diacritize(text: str, model_path: str | None = None) -> str:
    """Restore niqqud on `text`. Uses the bundled model by default (override with
    `model_path` or the NAKDIMON_MODEL environment variable). Diacritizer instances
    are cached per resolved model path, so repeated calls do not reload the model.
    """
    if model_path is None:
        model_path = default_model_path()
    diacritizer = _cache.get(model_path)
    if diacritizer is None:
        diacritizer = _cache[model_path] = Diacritizer(model_path)
    return diacritizer.diacritize(text)


__all__ = ["Diacritizer", "__version__", "default_model_path", "diacritize"]
