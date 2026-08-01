"""Capture v1 metrics as a golden, while v1 is still alive (run before Phase 4).

HISTORICAL: requires the v1 package and the pre-NFC test tree — both gone after
Phase 4. Re-run from git history (branch v1-archive) if ever needed.

Pins macro-averaged DEC/CHA/WOR/VOC (+OOV) for every system folder present in
tests/new, with the OOV vocabulary built by the v1 MajMod recipe (majority
dictionary over hebrew_diacritized/modern — full undotted words AND single
letters). The v2 metrics port must reproduce these numbers exactly on the
legacy-ordered test files.

    uv run python spec/tools/generate_metrics_golden.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from nakdimon import external_apis, metrics  # noqa: E402

SYSTEMS = ["Nakdimon", "Nakdimon_v2", "Dicta", "Morfix", "Snopi", "MajMod", "MajAllWithDicta"]


def main() -> None:
    possibilities: defaultdict[str, Counter] = defaultdict(Counter)
    external_apis.MajorityDiacritizer.update_possibilities(
        possibilities, ("hebrew_diacritized/modern",)
    )
    vocabulary = external_apis.MajorityDiacritizer(possibilities)

    stats = metrics.Stats(basepath=Path("tests/new/expected"), vocabulary=vocabulary)

    out: dict[str, object] = {
        "comment": "v1 macro-averaged metrics on tests/new; OOV vocabulary = "
                   "majority dictionary over hebrew_diacritized/modern (v1 MajMod "
                   "recipe: undotted full words + single letters)",
        "test_set": "tests/new",
        "vocabulary_paths": ["hebrew_diacritized/modern"],
        "vocabulary_size": len(vocabulary.dictionary),
        "systems": {},
    }

    for system in SYSTEMS:
        if not Path(f"tests/new/{system}").is_dir():
            continue
        out["systems"][system] = stats.macro_average(system)
        print(system, {k: f"{v:.4%}" for k, v in out["systems"][system].items()})

    # Per-folder means for one system, to pin the intermediate aggregation level.
    out["per_folder_Nakdimon"] = {
        folder_packs[0].source: stats.metricwise_mean(
            stats.all_metrics(doc_pack) for doc_pack in folder_packs
        )
        for folder_packs in stats.iter_documents_by_folder(["expected", "Nakdimon"])
    }

    dest = REPO_ROOT / "spec" / "golden" / "metrics_v1.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
