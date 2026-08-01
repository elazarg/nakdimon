"""The packaged spec (src/nakdimon/data/hebrew.json) must be an exact copy of the
repo's spec/hebrew.json — it is the single source of truth all ports read from."""

from __future__ import annotations

import json
from pathlib import Path

from nakdimon.spec import load_spec

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_packaged_spec_matches_repo_spec() -> None:
    repo_spec = json.loads((REPO_ROOT / "spec" / "hebrew.json").read_text(encoding="utf-8"))
    assert load_spec() == repo_spec


def test_packaged_spec_bytes_match_repo_spec() -> None:
    repo_bytes = (REPO_ROOT / "spec" / "hebrew.json").read_bytes()
    packaged_bytes = (Path(__file__).resolve().parents[1] / "src" / "nakdimon" / "data" / "hebrew.json").read_bytes()
    assert packaged_bytes == repo_bytes
