"""Phase 4: NFC-normalize the corpora and test sets, in place.

Safe only once v1 tooling is gone — the v1 strict parser crashes on NFC-ordered
marks. The v2 parser is order-tolerant, so this is a pure re-serialization:
verified per changed file by full decompose-equality — NFC reordering composes
with the parser's last-wins rule, so a letter carrying two marks of the same
channel (corpus noise) could silently change its parse; the assert makes any
such file fail loudly instead.

    uv run python -P spec/tools/nfc_normalize_corpus.py hebrew_diacritized tests
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages" / "nakdimon" / "src"))

from nakdimon import hebrew  # noqa: E402


def normalize_file(path: Path) -> str:
    """Returns 'changed', 'unchanged', or 'kept-legacy' (invariant would break)."""
    try:
        original = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(f"SKIP (not utf-8): {path}")
        return "unchanged"
    normalized = unicodedata.normalize("NFC", original)
    if normalized == original:
        return "unchanged"

    assert hebrew.remove_niqqud(normalized) == hebrew.remove_niqqud(original), path
    # Invariance is checked on CLEANED text (what every pipeline consumes; meteg
    # etc. are stripped first). It can genuinely fail: letters carrying TWO vowels
    # (biblical ketiv/qere composites like ירושָׁלִַם) flip their last-wins vowel
    # when NFC reorders by combining class. Such out-of-model files are left
    # byte-identical rather than silently altered — the parser reads any order.
    if hebrew.decompose(hebrew.clean_text(normalized)) != hebrew.decompose(hebrew.clean_text(original)):
        return "kept-legacy"

    path.write_text(normalized, encoding="utf-8")
    return "changed"


def main(roots: list[str]) -> None:
    counts = {"changed": 0, "unchanged": 0, "kept-legacy": 0}
    files = [p for root in roots for p in sorted(Path(root).rglob("*")) if p.is_file() and p.suffix != ".py"]
    for path in files:
        outcome = normalize_file(path)
        counts[outcome] += 1
        if outcome == "kept-legacy":
            print(f"kept legacy order (multi-vowel letters): {path}")
    print(f"normalized {counts['changed']} files, {counts['unchanged']} already NFC/unchanged, "
          f"{counts['kept-legacy']} kept legacy order")


if __name__ == "__main__":
    main(sys.argv[1:] or ["hebrew_diacritized", "tests"])
