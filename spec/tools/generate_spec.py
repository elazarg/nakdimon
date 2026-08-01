"""Generate spec/hebrew.json and spec/golden/*.json from the v1 codebase.

HISTORICAL: requires the v1 package (repo-root ./nakdimon), which was deleted in
Phase 4, and pre-NFC corpus files. To re-run, check out a pre-v2 revision from
git history (branch v1-archive). The generated spec and goldens are committed
and frozen; this file documents their provenance.

The v1 package (repo-root ./nakdimon) is used as the ORACLE: every table and
behavior captured here is asserted against the old code, so the spec is a
faithful description of what the shipped model was trained against.

The spec also records deliberate v2 decisions (inventory "v2", Nd digit rule)
where they diverge from v1; each divergence is listed explicitly.

Run from the repo root:

    uv run python spec/tools/generate_spec.py
"""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from nakdimon import dataset, hebrew, predict  # noqa: E402  (oracle imports)

SPEC_DIR = REPO_ROOT / "spec"
GOLDEN_DIR = SPEC_DIR / "golden"

QAMATS_QATAN = "ׇ"
HOLAM_HASER_FOR_VAV = "ֺ"
METEG = "ֽ"
CGJ = "͏"


# --------------------------------------------------------------------------
# Normalization: express the v1 normalize() as explicit data.
# --------------------------------------------------------------------------

def build_normalization() -> dict:
    # NOTE: final forms (ךםןףץ) are inside VALID_LETTERS, so they pass through
    # unchanged; v1's ENDINGS_TO_REGULAR branch was dead code. Finals are
    # first-class members of the letters vocabulary.
    explicit_map: dict[str, str] = {}
    for c in ["\n", "\t"]:
        explicit_map[c] = " "
    for c in ["־", "‒", "–", "—", "―", "−"]:
        explicit_map[c] = "-"
    explicit_map["["] = "("
    explicit_map["]"] = ")"
    for c in ["´", "‘", "’"]:
        explicit_map[c] = "'"
    for c in ["“", "”", "״"]:
        explicit_map[c] = '"'
    explicit_map["…"] = ","
    for c in ["ײ", "װ", "ױ"]:
        explicit_map[c] = "H"

    # Every explicit mapping and pass-through must agree with the oracle.
    for c, v in explicit_map.items():
        assert hebrew.normalize(c) == v, (c, v, hebrew.normalize(c))
    for c in hebrew.VALID_LETTERS:
        assert hebrew.normalize(c) == c
    for c in "0123456789٣٤۷":  # Nd examples, incl. Arabic-Indic
        assert hebrew.normalize(c) == "5"

    return {
        "pass_through": hebrew.VALID_LETTERS,
        "map": explicit_map,
        "digit_rule": {
            "comment": "any char in Unicode category Nd maps to the digit token",
            "category": "Nd",
            "to": "5",
        },
        "fallback": "O",
        "known_divergences_from_v1": [
            "v1 used Python str.isdigit(), which also matches some No/Lo chars "
            "(e.g. '²', '½' is False but '²' True); v2 restricts the digit rule "
            "to category Nd, so '²' now normalizes to 'O' instead of '5'.",
        ],
    }


def spec_normalize_char(c: str, norm: dict) -> str:
    """Reference implementation of the v2 normalization spec."""
    if c in norm["_pass"]:
        return c
    if c in norm["map"]:
        return norm["map"][c]
    if unicodedata.category(c) == "Nd":
        return "5"
    return "O"


def spec_normalize(text: str, norm: dict) -> str:
    return "".join(spec_normalize_char(c, norm) for c in text)


# --------------------------------------------------------------------------
# Inventories
# --------------------------------------------------------------------------

def encode_overrides(chars: list[str]) -> dict[str, int]:
    """Chars appearing more than once decode from their first index but ENCODE
    to their last (dict-overwrite in v1 CharacterTable). The v1 model was
    trained with the *encode* ids, so this asymmetry is load-bearing."""
    first: dict[str, int] = {}
    overrides: dict[str, int] = {}
    for i, c in enumerate(chars):
        if c in first:
            overrides[c] = i
        else:
            first[c] = i
    # sanity: mirror the oracle's encode dict
    for c, i in overrides.items():
        table = dataset.niqqud_table if c in dataset.niqqud_table.chars else None
        if table:
            assert table.char_indices[c] == i
    return overrides


def build_inventories() -> dict:
    v1_niqqud = list(dataset.niqqud_table.chars)
    v1 = {
        "letters": list(dataset.letters_table.chars),
        "niqqud": {
            "decode": v1_niqqud,
            "encode_overrides": encode_overrides(v1_niqqud),
        },
        "dagesh": {"decode": list(dataset.dagesh_table.chars)},
        "sin": {"decode": list(dataset.sin_table.chars)},
        "notes": [
            "index 0 is the mask/padding token (empty string)",
            "RAFE (U+05BF) = 'category applies but deliberately absent'",
            "niqqud contains PATAH twice (historic bug): decode index 9 and 15; "
            "encoding always produced 15, so the model predicts 15 for patah "
            "and index 9 is dead. Preserved for model compatibility.",
            "niqqud contains HOLAM HASER FOR VAV (U+05BA) although the corpus "
            "essentially never produces it.",
        ],
    }

    # v2: same letters; clean niqqud head. Single patah, drop U+05BA
    # (folded to holam in data prep), add qamats qatan U+05C7.
    v2_niqqud = [
        "",  # mask
        hebrew.RAFE,
        "ְ",  # shva
        "ֱ",  # hataf segol
        "ֲ",  # hataf patah
        "ֳ",  # hataf qamats
        "ִ",  # hiriq
        "ֵ",  # tsere
        "ֶ",  # segol
        "ַ",  # patah
        "ָ",  # qamats
        QAMATS_QATAN,
        "ֹ",  # holam
        "ֻ",  # qubuts
        "ּ",  # shuruk (on vav)
    ]
    v2 = {
        "letters": list(dataset.letters_table.chars),
        "niqqud": {"decode": v2_niqqud, "encode_overrides": {}},
        "dagesh": {"decode": list(dataset.dagesh_table.chars)},
        "sin": {"decode": list(dataset.sin_table.chars)},
        "data_folds": {
            HOLAM_HASER_FOR_VAV: "ֹ",
        },
        "notes": [
            "qamats qatan (U+05C7) is a first-class niqqud label; corpora must "
            "be relabeled (e.g. Dicta keepqq=true) before it carries signal. "
            "Until then training may fold it into qamats.",
            "models declare their inventory in the sidecar model card; "
            "the runtime must not assume v2.",
        ],
    }
    return {"v1": v1, "v2": v2}


# --------------------------------------------------------------------------
# Assemble spec
# --------------------------------------------------------------------------

def assert_ccc_ordering() -> None:
    """The NFC-emission claim rests on these combining classes; check, don't trust."""
    for c in "ְֱֲֳִֵֶַָֹֻ":  # vowels
        assert 10 <= unicodedata.combining(c) <= 20, (c, unicodedata.combining(c))
    assert unicodedata.combining(QAMATS_QATAN) == 18
    assert unicodedata.combining(HOLAM_HASER_FOR_VAV) == 19
    assert unicodedata.combining("ּ") == 21  # dagesh/shuruk
    assert unicodedata.combining(hebrew.RAFE) == 23
    assert unicodedata.combining("ׁ") == 24 and unicodedata.combining("ׂ") == 25
    # NFC on Hebrew letter+marks is a pure reorder (presentation forms are
    # composition exclusions): composing must never happen.
    assert unicodedata.normalize("NFC", "שּׁ") == "שּׁ" and len(unicodedata.normalize("NFC", "שּׁ")) == 3


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def items_to_text_nfc(items: list) -> str:
    """Reference v2 re-serialization: letter + niqqud + dagesh + sin is ascending
    combining-class order, hence an NFC fixpoint once RAFE is stripped."""
    txt = "".join(letter + niqqud + dagesh + sin for letter, _n, dagesh, sin, niqqud in items)
    txt = txt.replace(hebrew.RAFE, "")
    assert nfc(txt) == txt, repr(txt)
    return txt


def build_spec() -> dict:
    assert_ccc_ordering()
    return {
        "spec_version": "2.0.0",
        "oracle": "nakdimon 0.2.1 (repo-root package)",
        "mark_order": {
            "emit": "NFC",
            "emit_channel_order": ["niqqud", "dagesh", "sin"],
            "comment": (
                "v2 is NFC-native: emitted mark order is Unicode canonical — "
                "niqqud (ccc 10-20) before dagesh (ccc 21) before shin/sin dot "
                "(ccc 24/25). Hebrew presentation forms are composition "
                "exclusions, so NFC never composes letter+mark; it is a pure "
                "reorder and emitted text is an NFC fixpoint. Parsers must "
                "accept marks after a base letter in ANY order (last-wins per "
                "channel). NOTE: v1 emitted dagesh-sin-niqqud and its strict "
                "parser crashes on NFC-ordered text; legacy corpora stay in v1 "
                "order until v1 tooling is retired (DESIGN.md Phase 4)."
            ),
        },
        "special_tokens": {"foreign": "O", "ligature": "H", "digit": "5"},
        "normalization": build_normalization(),
        "inventories": build_inventories(),
        "decomposition": {
            "rafe": hebrew.RAFE,
            "can_dagesh": "בגדהוזטיכלמנספצקשתךף",
            "can_sin": "ש",
            "can_niqqud": "אבגדהוזחטיכלמנסעפצקרשתךן",
            "channel_order_in_text": ["dagesh", "sin", "niqqud"],
            "vav_shuruk_rule": (
                "if letter is vav with dagesh and no niqqud, reinterpret: "
                "dagesh := RAFE, niqqud := U+05BC (shuruk)"
            ),
        },
        "text_cleanup": {
            "comment": "chars deleted from input before decomposition/encoding",
            "strip": [METEG, CGJ, "‍", "‎", "‏", "﻿"],
        },
        "remove_niqqud": {
            "comment": "regex character class of chars removed by remove_niqqud",
            "chars": "ְ-ׇּֿׁׂ",
        },
        "output_convention": (
            "full-script (ktiv male) Hebrew with diacritics layered onto the "
            "unchanged input letter sequence, emitted in NFC canonical mark "
            "order; remove_niqqud(output) round-trips to the input (modulo "
            "whitespace collapsing). This is NOT normative ktiv-haser nikud; "
            "teachers/corpora must be projected into this convention."
        ),
    }


# --------------------------------------------------------------------------
# Golden vectors
# --------------------------------------------------------------------------

SAMPLES_UNDOTTED = [
    "שלום עולם",
    "בשנת 2024 קניתי 3 ספרים בגרמניה!",
    'צה"ל הודיע: [עדכון] "חדש" — ראו ־ כאן…',
    "ךםןףץ סופיות באמצע",
    "hello עברית mixed with ٣ and ² and ½",
    "שורה\nחדשה\tוטאב",
    "װאס ױ ײדיש",
    "אין לי מושג מה יקרה מחר, אבל זה בסדר גמור.",
]

DAGESH = "ּ"      # U+05BC
SHIN = "ׁ"        # U+05C1
SIN = "ׂ"         # U+05C2
SHVA, HATAF_PATAH, HIRIQ, TSERE, SEGOL, PATAH, QAMATS, HOLAM, QUBUTS = (
    "ְ", "ֲ", "ִ", "ֵ", "ֶ", "ַ", "ָ", "ֹ", "ֻ"
)


def _w(*groups: str) -> str:
    """Build a dotted word from per-letter groups written in channel order."""
    return "".join(groups)


# Synthetic edge cases, constructed in the corpus channel order
# (letter, dagesh?, sin-dot?, niqqud?) so they are well-formed by construction.
SAMPLES_DOTTED = [
    # shin-dot + niqqud, vav-holam
    _w("ש" + SHIN + QAMATS, "ל", "ו" + HOLAM, "ם") + " " +
    _w("ע", "ו" + HOLAM, "ל" + QAMATS, "ם"),
    # dagesh+niqqud on same letter, shuruk as vav+dagesh, sin-smalit
    _w("מ" + SHVA, "ד" + QUBUTS, "ב" + DAGESH + QAMATS, "ר") + " " +
    _w("ו" + DAGESH, "ב" + SHVA, "ש" + SIN + QAMATS, "ר" + QAMATS, "ה") + " " +
    _w("ב" + DAGESH + SHVA, "ע" + HIRIQ, "ב" + SHVA, "ר" + HIRIQ, "י", "ת"),
    # final kaf with shva, hataf, gershayim acronym left undotted
    _w("ה" + PATAH, "י" + DAGESH + SEGOL, "ל" + SEGOL, "ד") + " " +
    _w("ח" + HATAF_PATAH, "ב" + TSERE, "ר", "ו" + HOLAM) + " " +
    _w("ש" + SHIN + SEGOL, "ל" + QAMATS, "ך" + SHVA) + ' צה"ל',
]


def corpus_snippets() -> list[str]:
    """Deterministic real-corpus dotted snippets from the test sets."""
    out = []
    base = REPO_ROOT / "tests" / "new" / "expected"
    files = sorted(p for p in base.rglob("*") if p.is_file())[:3]
    for p in files:
        text = " ".join(p.read_text(encoding="utf8").split())
        out.append(text[:280].rsplit(" ", 1)[0])
    return out


def build_golden(spec: dict) -> dict[str, object]:
    norm = dict(spec["normalization"])
    norm["_pass"] = set(norm["pass_through"])

    dotted = SAMPLES_DOTTED + corpus_snippets()
    undotted = SAMPLES_UNDOTTED + [hebrew.remove_niqqud(t) for t in dotted]

    normalize_vec = [
        {"input": t, "normalized": spec_normalize(t, norm)} for t in undotted
    ]

    encode_vec = []
    for t in undotted:
        flat = spec_normalize(t.replace("\n", " ").replace("\t", " "), norm)
        ids = dataset.letters_table.to_ids([flat])[0]
        encode_vec.append({"text": t, "normalized_flat": flat, "ids": ids})

    decompose_vec = []
    for t in dotted:
        items = list(hebrew.iterate_dotted_text(t))
        tuples = [[i.letter, i.normalized, i.dagesh, i.sin, i.niqqud] for i in items]
        decompose_vec.append({
            "dotted": t,                # legacy v1 mark order (as in the corpora)
            "dotted_nfc": nfc(t),       # canonical order; MUST decompose identically
            "items": tuples,
            "reconstructed": items_to_text_nfc(tuples),  # v2 emission: NFC fixpoint
            "niqqud_ids": dataset.niqqud_table.to_ids([[i.niqqud for i in items]])[0],
            "dagesh_ids": dataset.dagesh_table.to_ids([[i.dagesh for i in items]])[0],
            "sin_ids": dataset.sin_table.to_ids([[i.sin for i in items]])[0],
        })

    remove_vec = []
    for t in dotted:
        stripped = hebrew.remove_niqqud(t)
        assert hebrew.remove_niqqud(nfc(t)) == stripped  # order-insensitive
        remove_vec.append({"input": t, "input_nfc": nfc(t), "output": stripped})

    model_path = REPO_ROOT / "nakdimon" / "data" / "Nakdimon.onnx"
    predict_vec = {
        "model": "nakdimon/data/Nakdimon.onnx",
        "comment": "outputs of the v1 predict() pipeline stored in NFC (v2 "
                   "runtimes emit NFC natively, so these must match byte-exactly; "
                   "comparing against raw v1 output requires NFC on both sides)",
        "cases": [
            {"input": t, "output": nfc(predict.predict(t, str(model_path)))}
            for t in [hebrew.remove_niqqud(t) for t in dotted[:4]]
            + ["שלום עולם", "אין לי מושג מה יקרה מחר, אבל זה בסדר גמור."]
        ],
    }

    return {
        "normalize.json": normalize_vec,
        "encode_v1.json": encode_vec,
        "decompose.json": decompose_vec,
        "remove_niqqud.json": remove_vec,
        "predict_v1.json": predict_vec,
    }


def main() -> None:
    spec = build_spec()
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    (SPEC_DIR / "hebrew.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=1) + "\n", encoding="utf8"
    )
    for name, data in build_golden(spec).items():
        (GOLDEN_DIR / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf8"
        )
    print(f"wrote {SPEC_DIR / 'hebrew.json'}")
    for name in sorted(p.name for p in GOLDEN_DIR.iterdir()):
        print(f"wrote {GOLDEN_DIR / name}")


if __name__ == "__main__":
    main()
