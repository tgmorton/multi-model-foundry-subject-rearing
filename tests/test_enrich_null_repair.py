"""The v2 stimulus roots: enrichment null items keep the overt item's verb agreement,
and nothing else changes relative to the vetted v1 roots."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from repair_enrich_null_agreement import CONDITION, by_pair, is_overt_minus_one_word, read_rows  # noqa: E402

ST = REPO / "evaluation/stimuli"
ROOTS = [("null-subj-v2-matched-v1", "null-subj-v2-matched-v2"),
         ("null-subj-v2-matched-nopron-v1", "null-subj-v2-matched-nopron-v2"),
         ("null-subj-v2-priming-nopron-v1", "null-subj-v2-priming-nopron-v2"),
         ("null-subj-v2-nopron-plus-priming-v1", "null-subj-v2-nopron-plus-priming-v2")]
OTHERS = ["baseline", "impoverish_case", "lemmatize_verbs", "remove_expletive_sentences"]


def enrich_rows(root: str) -> list[dict[str, str]]:
    return [r for f in sorted((ST / root / CONDITION / "en").glob("*.csv")) for r in read_rows(f)]


@pytest.mark.parametrize("v1,v2", ROOTS)
def test_other_conditions_are_byte_identical(v1, v2):
    for cond in OTHERS:
        names = sorted(p.name for p in (ST / v1 / cond / "en").glob("*.csv"))
        assert names == sorted(p.name for p in (ST / v2 / cond / "en").glob("*.csv"))
        for name in names:
            assert (ST / v1 / cond / "en" / name).read_bytes() == (ST / v2 / cond / "en" / name).read_bytes()


@pytest.mark.parametrize("v1,v2", ROOTS)
def test_root_passes_the_eval_validator(v1, v2):
    pytest.importorskip("torch")
    import eval_v2_cell as E
    manifest, _ = E._validate_matched_stimuli(ST / v2, None)
    assert manifest["vetted"] is True


@pytest.mark.parametrize("root", [v2 for _, v2 in ROOTS])
def test_every_enrichment_pair_is_overt_minus_one_word(root):
    pairs = by_pair(enrich_rows(root))
    assert pairs
    for key, m in pairs.items():
        assert is_overt_minus_one_word(m["1"]["target"], m["0"]["target"]), key
        words = m["0"]["target"].split()
        assert m["0"]["hotspot_token"] == words[int(m["0"]["hotspot_position"])], key


@pytest.mark.parametrize("v1,v2", ROOTS[:2])
def test_repair_touches_only_null_targets(v1, v2):
    old, new = by_pair(enrich_rows(v1)), by_pair(enrich_rows(v2))
    assert old.keys() == new.keys()
    changed = 0
    for key in old:
        assert old[key]["1"] == new[key]["1"], key
        a = {k: v for k, v in old[key]["0"].items() if k not in ("target", "hotspot_token")}
        b = {k: v for k, v in new[key]["0"].items() if k not in ("target", "hotspot_token")}
        assert a == b, key
        changed += old[key]["0"]["target"] != new[key]["0"]["target"]
    assert changed == 256


def test_repair_keeps_the_agreement_suffix():
    pairs = by_pair(enrich_rows("null-subj-v2-matched-v2"))
    m = pairs[("subject_drop", "subj_1pl", "2")]
    assert m["1"]["target"] == "we cheerimus for the home team ."
    assert m["0"]["target"] == "cheerimus for the home team ."


def test_priming_pairing_is_unchanged():
    a = (ST / "null-subj-v2-priming-nopron-v1/pairing.csv").read_bytes()
    assert a == (ST / "null-subj-v2-priming-nopron-v2/pairing.csv").read_bytes()


def test_priming_rebuild_reproduces_v1_exactly_outside_the_repair():
    """Overt-prime rows of a target whose null form did not change must be unchanged."""
    with (ST / "null-subj-v2-priming-nopron-v2/pairing.csv").open(newline="", encoding="utf-8") as fh:
        n_items = sum(1 for _ in csv.DictReader(fh))
    assert n_items == 9216
    old, new = by_pair(enrich_rows("null-subj-v2-priming-nopron-v1")), by_pair(enrich_rows("null-subj-v2-priming-nopron-v2"))
    assert old.keys() == new.keys()
    for key in old:
        if key[1].endswith("__overt"):
            assert old[key]["1"]["context"] == new[key]["1"]["context"], key
            assert old[key]["1"]["target"] == new[key]["1"]["target"], key
