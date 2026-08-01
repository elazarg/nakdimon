"""Evaluation entry point for the v2 metrics stack: diacritize a test set with an ONNX
model and/or report macro-averaged DEC/CHA/WOR/VOC(+OOV) metrics for one or more
systems already present as sibling folders of a test set's `expected/` directory.

Mirrors nakdimon/run_test.py (diacritize_all) + nakdimon/metrics.py (main), minus the
external-API systems (Snopi/Morfix/Dicta fetchers, which need `requests` and network
access) -- those are a research extra elsewhere, not this module's concern. No torch
import: only the v2 runtime (onnxruntime) + numpy are required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nakdimon import hebrew
from nakdimon.runtime import Diacritizer

from nakdimon_train.metrics import MajorityVocabulary, Stats

DEFAULT_VOCABULARY_PATHS = ["hebrew_diacritized/modern"]


def run_system(
    model_path: str | Path,
    test_set_dir: str | Path,
    out_name: str | None = None,
    skip_existing: bool = False,
) -> None:
    """Diacritize every file under `{test_set_dir}/expected` with the ONNX model at
    `model_path`, writing results to the sibling folder `{test_set_dir}/{out_name}`
    (default: the model file's stem, e.g. 'Nakdimon' for '.../Nakdimon.onnx') --
    mirrors v1's run_test.diacritize_all path structure (swap the 'expected' path
    segment for the system name).
    """
    model_path = Path(model_path)
    name = out_name or model_path.stem
    diacritizer = Diacritizer(model_path)

    expected_dir = Path(test_set_dir) / "expected"
    for infile in sorted(expected_dir.rglob("*")):
        if not infile.is_file():
            continue
        outfile = Path(str(infile).replace("expected", name))
        if outfile.exists():
            if skip_existing:
                continue
            outfile.unlink()
        outfile.parent.mkdir(parents=True, exist_ok=True)
        expected_text = infile.read_text(encoding="utf8")
        cleaned = hebrew.remove_niqqud(expected_text)
        actual = diacritizer.diacritize(cleaned)
        outfile.write_text(actual, encoding="utf8")


def evaluate(
    test_set_dir: str | Path,
    systems: list[str],
    vocabulary_paths: list[str | Path],
) -> dict[str, dict[str, float]]:
    """Macro-average DEC/CHA/WOR/VOC(+OOV) for each of `systems` against
    `{test_set_dir}/expected`, using an OOV vocabulary built from `vocabulary_paths`.
    """
    vocabulary = MajorityVocabulary.from_paths([Path(p) for p in vocabulary_paths])
    stats = Stats(basepath=Path(test_set_dir) / "expected", vocabulary=vocabulary)
    return {system: stats.macro_average(system) for system in systems}


def _autodiscover_systems(test_set_dir: str | Path) -> list[str]:
    base = Path(test_set_dir)
    return sorted(p.name for p in base.iterdir() if p.is_dir() and p.name != "expected")


def _check_leakage_guard(test_set: str, systems: list[str]) -> None:
    """Refuse to evaluate systems built from (or trained on) the dicta test set
    against that same test set -- port of v1's rule in nakdimon/metrics.py main(),
    which silently dropped 'Nakdimon' and 'MajAllWithDicta' when test_set was exactly
    'tests/dicta'. Here we raise instead of silently dropping, and match by test-set
    *suffix* / 'Nakdimon' *substring* so this also covers e.g. 'tests/dicta_v2' and
    model variants like 'Nakdimon_v2'.
    """
    if not test_set.rstrip("/").endswith("dicta"):
        return
    blocked = [s for s in systems if s == "MajAllWithDicta" or "Nakdimon" in s]
    if blocked:
        raise SystemExit(
            f"Refusing to evaluate {blocked} against {test_set!r}: these systems were "
            "trained on, or built from, the dicta test set, so evaluating them on it "
            "would leak train/test data. Remove them from --systems or use a "
            "dicta-free vocabulary/model."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate diacritization systems against a test set (v2 metrics port)."
    )
    parser.add_argument("--test-set", default="tests/new", help="test set directory (must contain 'expected/')")
    parser.add_argument(
        "--systems",
        nargs="+",
        default=None,
        help="systems to evaluate (default: autodiscover sibling folders of 'expected')",
    )
    parser.add_argument(
        "--vocabulary",
        nargs="+",
        default=DEFAULT_VOCABULARY_PATHS,
        help="corpus paths used to build the OOV majority vocabulary",
    )
    args = parser.parse_args()

    systems = args.systems if args.systems is not None else _autodiscover_systems(args.test_set)
    _check_leakage_guard(args.test_set, systems)

    results = evaluate(args.test_set, systems, args.vocabulary)
    for system, system_metrics in results.items():
        print(system, {k: f"{v:.4%}" for k, v in system_metrics.items()})


if __name__ == "__main__":
    main()
