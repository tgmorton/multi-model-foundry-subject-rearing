from pathlib import Path

import pytest

from scripts.generate_condition_matched_stimuli import (
    SCHEMA,
    reconcile_shared_edits,
    shared_edit_divergences,
    validate_condition,
)


def _row(status: int, target: str, hotspot_position: int = 1):
    words = target.split()
    return dict(zip(SCHEMA, [
        "1", "subject_drop", "subj_1pl", str(status), "we won .",
        target, words[hotspot_position], str(hotspot_position), "en", "",
        "test",
    ]))


def test_reconcile_copies_overt_enrichment_to_aligned_null_verb():
    target, forced = reconcile_shared_edits(
        "we walk home .", "walk home .",
        "we walkamus home .", "walkakt home .",
    )
    assert target == "walkamus home ."
    assert forced == [{
        "source_token": "walk", "overt_position": 1, "null_position": 0,
        "independent_null": "walkakt", "reconciled": "walkamus",
    }]


def test_literal_enrichment_divergence_is_reported():
    divergences = shared_edit_divergences(
        "we walk home .", "walk home .",
        "we walkamus home .", "walkakt home .",
    )
    assert len(divergences) == 1
    assert divergences[0]["source_token"] == "walk"
    assert divergences[0]["overt_transformed"] == "walkamus"
    assert divergences[0]["null_transformed"] == "walkakt"


def test_reconcile_refuses_token_count_changes():
    with pytest.raises(ValueError, match="token-count-changing"):
        reconcile_shared_edits("we walk .", "walk .", "we do walk .", "walk .")


def test_structural_validator_requires_complete_pairs():
    path = Path("subject_drop.csv")
    outputs = {path: [_row(1, "we walkamus home .", 1)]}
    with pytest.raises(ValueError, match="broken pair"):
        validate_condition("enrich_verbal_morphology", 2, outputs, [{
            "excluded": False, "forced_shared_edits": [],
            "pair_divergent_shared_edits": [],
        }])
