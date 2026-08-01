"""Loader for the packaged copy of spec/hebrew.json — the single source of truth for
normalization, inventories, and decomposition rules shared across the Nakdimon ports."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any


@lru_cache(maxsize=1)
def load_spec() -> dict[str, Any]:
    text = files("nakdimon").joinpath("data", "hebrew.json").read_text(encoding="utf-8")
    return json.loads(text)
