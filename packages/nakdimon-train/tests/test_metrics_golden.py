"""Golden-vector test: the v2 metrics port must reproduce v1's macro-averaged
DEC/CHA/WOR/VOC(+OOV) on tests/new exactly. spec/golden/metrics_v1.json pins these
numbers, captured from v1 (nakdimon.metrics.Stats) while it was still the canonical
implementation -- see spec/tools/generate_metrics_golden.py for the recipe (OOV
vocabulary = v1 MajMod majority dictionary over hebrew_diacritized/modern: full
undotted words plus single letters).

Also covers two of the trickiest ported semantics in isolation: the niqqud
vocalization collapse table (tokens.vocalize_item) and the strict=False (truncate to
shorter) zip semantics that Stats.mean_equal relies on.
"""

from __future__ import annotations

import math

import pytest
from nakdimon.hebrew import HebrewItem

from nakdimon_train.metrics import Stats
from nakdimon_train.tokens import Token, vocalize

ABS_TOL = 1e-9

SYSTEMS = ["Nakdimon", "Nakdimon_v2", "Dicta", "Morfix", "Snopi", "MajMod", "MajAllWithDicta"]


def _assert_metrics_close(actual: dict[str, float], expected: dict[str, float], label: str) -> None:
    assert set(actual) == set(expected), f"{label}: metric key mismatch {set(actual)} != {set(expected)}"
    for key, expected_value in expected.items():
        actual_value = actual[key]
        assert math.isclose(actual_value, expected_value, rel_tol=0.0, abs_tol=ABS_TOL), (
            f"{label}.{key}: {actual_value!r} != {expected_value!r} "
            f"(abs diff {abs(actual_value - expected_value):.3e})"
        )


from conftest import requires_pre_nfc_tree


@requires_pre_nfc_tree
def test_vocabulary_size_matches_v1(vocabulary, golden) -> None:
    assert len(vocabulary.dictionary) == golden["vocabulary_size"]


@requires_pre_nfc_tree
@pytest.mark.parametrize("system", SYSTEMS)
def test_macro_average_matches_v1(stats: Stats, golden: dict, system: str) -> None:
    expected = golden["systems"][system]
    actual = stats.macro_average(system)
    _assert_metrics_close(actual, expected, system)


@requires_pre_nfc_tree
def test_per_folder_nakdimon_matches_v1(stats: Stats, golden: dict) -> None:
    expected_per_folder = golden["per_folder_Nakdimon"]
    actual_per_folder = {
        folder_packs[0].source: stats.metricwise_mean(stats.all_metrics(doc_pack) for doc_pack in folder_packs)
        for folder_packs in stats.iter_documents_by_folder(["expected", "Nakdimon"])
    }
    assert set(actual_per_folder) == set(expected_per_folder)
    for folder, expected_metrics in expected_per_folder.items():
        _assert_metrics_close(actual_per_folder[folder], expected_metrics, folder)


# --- v2 golden: the going-forward pin (order-tolerant parser, sorted iteration) ----
# These numbers are NFC-invariant by construction, so they must not move when the
# test tree is normalized (Phase 4) -- that invariance is itself part of the test.


@pytest.mark.parametrize("system", SYSTEMS)
def test_macro_average_matches_v2_golden(stats_v2: Stats, golden_v2: dict, system: str) -> None:
    _assert_metrics_close(stats_v2.macro_average(system), golden_v2["systems"][system], f"v2:{system}")


# --- Unit tests for two of the trickier ported traps -------------------------------


def _item(letter: str, niqqud: str = "", dagesh: str = "", sin: str = "") -> HebrewItem:
    return HebrewItem(letter=letter, normalized=letter, dagesh=dagesh, sin=sin, niqqud=niqqud)


@pytest.mark.parametrize(
    ("niqqud_in", "niqqud_out"),
    [
        ("ָ", "ַ"),  # kamatz -> patah
        ("ַ", "ַ"),  # patah -> patah
        ("ֲ", "ַ"),  # hataf-patah -> patah
        ("ֹ", "ֹ"),  # holam -> holam
        ("ֳ", "ֹ"),  # hataf-kamatz -> holam
        ("ּ", "ֻ"),  # shuruk -> kubutz
        ("ֻ", "ֻ"),  # kubutz -> kubutz
        ("ֵ", "ֶ"),  # tzeire -> segol
        ("ֶ", "ֶ"),  # segol -> segol
        ("ֱ", "ֶ"),  # hataf-segol -> segol
        ("ְ", ""),  # shva -> ''
        ("ֿ", ""),  # rafe (no decision) -> ''
    ],
)
def test_vocalize_niqqud_collapse_table(niqqud_in: str, niqqud_out: str) -> None:
    token = Token((_item("א", niqqud=niqqud_in),))
    assert vocalize(token).items[0].niqqud == niqqud_out


def test_vocalize_dagesh_only_survives_on_beged_kefet() -> None:
    # Dagesh on a beged-kefet letter (בכפ) is a real vocalization choice and survives.
    beged = Token((_item("ב", dagesh="ּ"),))
    assert vocalize(beged).items[0].dagesh == "ּ"

    # Dagesh recorded on any other letter (e.g. a doubling dagesh) is dropped: it
    # doesn't change the letter's pronunciation class the way beged-kefet dagesh does.
    other = Token((_item("א", dagesh="ּ"),))
    assert vocalize(other).items[0].dagesh == ""


def test_mean_equal_truncates_on_length_mismatch() -> None:
    """Stats.mean_equal pools whatever pairs its caller hands it; metric_* methods
    build those pairs with zip(actual, expected, strict=False), which silently
    truncates to the shorter sequence instead of raising on a length mismatch. This
    is load-bearing for the golden (actual/expected item and token counts can differ
    across systems) and must not be "fixed" to strict=True.
    """
    actual_items = [_item("א", niqqud="ָ"), _item("ב", niqqud="ַ"), _item("ג", niqqud="ְ")]
    expected_items = [_item("א", niqqud="ָ"), _item("ב", niqqud="ְ")]

    stats = Stats(basepath=None, vocabulary=None)
    pairs = zip(actual_items, expected_items, strict=False)
    # Only 2 pairs are compared (the third actual item has no partner and is dropped):
    # ("ָ" == "ָ") -> True, ("ַ" == "ְ") -> False.
    assert stats.mean_equal(pairs) == 0.5
