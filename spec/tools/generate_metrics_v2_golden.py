"""Generate spec/golden/metrics_v2.json: the going-forward metrics pin.

Computed BY the v2 implementation (nakdimon_train.metrics with its defaults:
order-tolerant parser, sorted iteration, mark folding). Its authority is
transitive: the same implementation reproduces the independent v1 oracle
(spec/golden/metrics_v1.json) to 0.0 deviation in legacy mode on the pre-NFC
tree; the delta between the two goldens is exactly the documented parser
differences (double-dagesh glitch forgiveness, sorted aggregation order).

Because the default parser is order-tolerant, these numbers are invariant under
NFC normalization of the test files — re-running after Phase 4 must reproduce
this file exactly.

    uv run python -P spec/tools/generate_metrics_v2_golden.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages" / "nakdimon" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "nakdimon-train" / "src"))

from nakdimon_train.metrics import MajorityVocabulary, Stats  # noqa: E402

SYSTEMS = ["Nakdimon", "Nakdimon_v2", "Dicta", "Morfix", "Snopi", "MajMod", "MajAllWithDicta"]


def main() -> None:
    vocabulary = MajorityVocabulary.from_paths([REPO_ROOT / "hebrew_diacritized" / "modern"])
    stats = Stats(basepath=REPO_ROOT / "tests" / "new" / "expected", vocabulary=vocabulary)

    out = {
        "comment": "v2 metrics defaults (order-tolerant parser, sorted iteration, "
                   "fold qq/holam-haser) on tests/new; OOV vocabulary = majority "
                   "dictionary over hebrew_diacritized/modern. NFC-invariant.",
        "vocabulary_size": len(vocabulary.dictionary),
        "systems": {system: stats.macro_average(system) for system in SYSTEMS},
    }

    dest = REPO_ROOT / "spec" / "golden" / "metrics_v2.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    print(f"wrote {dest}")
    for system, m in out["systems"].items():
        print(system, {k: f"{v:.4%}" for k, v in m.items()})


if __name__ == "__main__":
    main()
