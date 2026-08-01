"""Silver-data pipeline: label undotted modern text with a teacher, project the
output into the corpus convention, write silver corpus files (DESIGN.md §6.2).

Every teacher goes through the convention projector: mark-level normalization
plus alignment projection, with respelled tokens rejected (never repaired).
This is where the "LLMs produce nikud that is too good" constraint is enforced
mechanically — normative/ktiv-haser output simply fails projection.

Live-API use is intentionally minimal here: check Dicta ToS before bulk runs.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

import requests

from nakdimon import hebrew
from nakdimon_train.data.conventions import normalize_marks, project_text

Teacher = Callable[[str], str]

DICTA_URL = "https://nakdan-2-0.loadbalancer.dicta.org.il/api"


def dicta_teacher(genre: str = "modern", *, keep_qamats_qatan: bool = False, timeout: float = 60.0) -> Teacher:
    def teach(text: str) -> str:
        payload = {
            "task": "nakdan",
            "genre": genre,
            "data": text,
            "addmorph": False,
            "keepqq": keep_qamats_qatan,
            "nodageshdefmem": False,
            "patachma": False,
            "keepmetagim": True,
        }
        r = requests.post(
            DICTA_URL, json=payload, headers={"content-type": "text/plain;charset=UTF-8"}, timeout=timeout
        )
        r.raise_for_status()
        words = [k["options"][0][0] if k.get("options") else k["word"] for k in r.json()]
        return normalize_marks("".join(words), keep_qamats_qatan=keep_qamats_qatan)

    return teach


def label_file(path: Path, out_path: Path, teacher: Teacher) -> tuple[int, int]:
    """Teacher-label one undotted file; returns (kept, rejected) token counts.
    The projected text always satisfies the alignment contract, so a partially
    rejected document is still usable training data (rejected tokens stay bare)."""
    text = " ".join(path.read_text(encoding="utf-8").split())
    undotted = hebrew.remove_niqqud(hebrew.clean_text(text))
    projected, kept, rejected = project_text(undotted, teacher(undotted))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(projected + "\n", encoding="utf-8")
    return kept, rejected


def label_corpus(in_dir: str, out_dir: str, teacher: Teacher, *, delay_s: float = 1.0) -> None:
    total_kept = total_rejected = 0
    for path in sorted(Path(in_dir).rglob("*.txt")):
        out_path = Path(out_dir) / path.relative_to(in_dir)
        if out_path.exists():
            continue
        kept, rejected = label_file(path, out_path, teacher)
        total_kept += kept
        total_rejected += rejected
        logging.info(f"{path}: kept {kept}, rejected {rejected}")
        time.sleep(delay_s)
    logging.info(f"total: kept {total_kept}, rejected {total_rejected} "
                 f"({total_rejected / max(1, total_kept + total_rejected):.2%} rejection)")
