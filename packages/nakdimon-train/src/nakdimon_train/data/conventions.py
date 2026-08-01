"""Convention projection: force teacher output into the corpus register.

The corpus convention is full-script (ktiv male) spelling with marks layered
onto the letters as written (spec/hebrew.json "output_convention"). Teachers —
Dicta, and especially LLMs — drift toward normative nikud: ktiv-haser
respellings, meteg, qamats qatan everywhere. Their output must be projected
into the corpus convention, and tokens that would require changing the letter
sequence are REJECTED, never repaired: silver data with silent respellings is
worse than less silver data.

These are the mark-level rewrites v1 applied ad-hoc inside its Dicta fetcher,
promoted to a single named, testable transform shared by the silver pipeline,
the qamats-qatan relabeler, and external-system evaluation.
"""

from __future__ import annotations

import re

from nakdimon import hebrew

METEG = "ֽ"
MAQAF = "־"
QAMATS = "ָ"
QAMATS_QATAN = "ׇ"
HATAF_QAMATS = "ֳ"
HOLAM = "ֹ"
QUBUTS = "ֻ"
DAGESH = "ּ"
VAV = "ו"

_KAMATZ_VAV_BEFORE_LETTER = re.compile(QAMATS + VAV + "(?=[א-ת])")


def normalize_marks(text: str, *, keep_qamats_qatan: bool = False) -> str:
    """Project teacher mark conventions onto the corpus convention.

    Preserves the letter sequence by construction: every rewrite either drops
    marks or moves them between adjacent chars.
    """
    res = text.replace("|", "")
    res = res.replace(QUBUTS + VAV + METEG, VAV + DAGESH)  # shuruk written as u-vav
    res = res.replace(HOLAM + VAV + METEG, VAV + HOLAM)
    if not keep_qamats_qatan:
        res = res.replace(QAMATS_QATAN, QAMATS)
    res = res.replace(METEG, "")
    res = _KAMATZ_VAV_BEFORE_LETTER.sub(VAV + HOLAM, res)
    res = res.replace(HATAF_QAMATS + VAV, VAV + HOLAM)
    res = res.replace(DAGESH + DAGESH, DAGESH)
    res = res.replace(MAQAF, "-")
    return res


def project_token(gold_undotted: str, teacher_dotted: str) -> str | None:
    """Return the teacher's dotting of a token iff it dots exactly the gold
    letters; None means the teacher respelled the token (reject)."""
    if hebrew.remove_niqqud(teacher_dotted) != gold_undotted:
        return None
    return teacher_dotted


def project_text(gold_undotted: str, teacher_dotted: str) -> tuple[str, int, int]:
    """Token-wise projection of a whole teacher output onto gold letters.

    Returns (projected_text, kept, rejected); rejected tokens fall back to the
    undotted gold token, so the result always satisfies the alignment contract.
    """
    gold_tokens = gold_undotted.split()
    teacher_tokens = teacher_dotted.split()
    if len(gold_tokens) != len(teacher_tokens):
        # Teacher merged/split tokens: reject the whole document — token-level
        # realignment here would hide systematic teacher drift.
        return gold_undotted, 0, len(gold_tokens)
    out: list[str] = []
    kept = rejected = 0
    for gold, teacher in zip(gold_tokens, teacher_tokens, strict=True):
        projected = project_token(gold, teacher)
        if projected is None:
            out.append(gold)
            rejected += 1
        else:
            out.append(projected)
            kept += 1
    return " ".join(out), kept, rejected
