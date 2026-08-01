"""Qamats-qatan relabeling: upgrade gold corpora to the v2 inventory.

Existing corpora type qamats qatan (U+05C7) as plain qamats (U+05B8). The v2
inventory makes qq a first-class label (DESIGN.md §4), which is only trainable
after relabeling. Strategy: ask a qq-aware teacher (Dicta nakdan with
keepqq=true) to dot the undotted gold text, then adopt ONLY the
qamats→qamats-qatan substitutions — every other disagreement keeps gold.
This makes the relabel a strict, reviewable delta over the gold corpus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from nakdimon import hebrew
from nakdimon_train.data.conventions import QAMATS, QAMATS_QATAN


@dataclass(frozen=True)
class QQSubstitution:
    token_index: int
    gold: str
    relabeled: str


def qq_teacher():
    """Dicta nakdan with keepqq=true — the one-flag flip that makes the teacher
    emit U+05C7. Untested against the live API; verify interactively before a
    bulk run (and check ToS for bulk labeling)."""
    from nakdimon_train.data.silver import dicta_teacher

    return dicta_teacher(keep_qamats_qatan=True)


def adopt_qq_only(gold_token: str, teacher_token: str) -> str | None:
    """Return gold_token with qq marks adopted from the teacher, or None if the
    tokens differ by anything other than qamats→qamats-qatan."""
    gold_items = hebrew.decompose(gold_token)
    teacher_items = hebrew.decompose(teacher_token)
    if len(gold_items) != len(teacher_items):
        return None
    out = []
    changed = False
    for g, t in zip(gold_items, teacher_items, strict=True):
        if (g.letter, g.dagesh, g.sin) != (t.letter, t.dagesh, t.sin):
            return None
        if g.niqqud == t.niqqud:
            out.append(g)
        elif g.niqqud == QAMATS and t.niqqud == QAMATS_QATAN:
            out.append(g._replace(niqqud=QAMATS_QATAN))
            changed = True
        else:
            return None
    return hebrew.items_to_text(out) if changed else gold_token


def relabel_document(gold_dotted: str, teacher_dotted: str) -> tuple[str, list[QQSubstitution]]:
    """Relabel one document; anything unadoptable silently keeps gold, and
    every adopted substitution is reported for review."""
    gold_tokens = gold_dotted.split()
    teacher_tokens = teacher_dotted.split()
    subs: list[QQSubstitution] = []
    if len(gold_tokens) != len(teacher_tokens):
        return gold_dotted, subs
    out = []
    for i, (gold, teacher) in enumerate(zip(gold_tokens, teacher_tokens, strict=True)):
        adopted = adopt_qq_only(gold, teacher)
        if adopted is not None and adopted != gold:
            subs.append(QQSubstitution(i, gold, adopted))
            out.append(adopted)
        else:
            out.append(gold)
    return " ".join(out), subs


def substitution_report(subs: list[QQSubstitution]) -> str:
    """One JSON line per substitution — the unit of manual review."""
    return "\n".join(
        json.dumps({"i": s.token_index, "gold": s.gold, "qq": s.relabeled}, ensure_ascii=False) for s in subs
    )
