"""Make sure tests import the src-layout package in this directory, never the
repo-root v1 `nakdimon` package (which is not on the path by default, but a
misconfigured environment could still shadow this one)."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
