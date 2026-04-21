"""D1 test: v2 stimulus schema loading.

Asserts that the NullSubjectEvaluator.load_stimuli_file method correctly:
- accepts the v2 schema produced by the new stimuli pipeline,
- normalizes columns to the internal v1 naming,
- preserves v2-only fields (condition, language),
- validates hotspot_position → target.split()[hotspot_position] == hotspot_token,
- still accepts v1 schema (backwards compatibility).
"""

from pathlib import Path

import pandas as pd
import pytest

from evaluation.evaluators.null_subject_evaluator import NullSubjectEvaluator


REPO_ROOT = Path(__file__).resolve().parents[2]
V2_EN_DIR = REPO_ROOT / "evaluation" / "stimuli" / "null-subj-v2" / "staging" / "en"
V2_ES_DIR = REPO_ROOT / "evaluation" / "stimuli" / "null-subj-v2" / "staging" / "es"
V1_DIR = REPO_ROOT / "evaluation" / "stimuli" / "null-subj"


def _make_evaluator():
    # load_stimuli_file doesn't touch the surprisal calculator; a stub is fine.
    class _Stub:
        pass
    ev = NullSubjectEvaluator.__new__(NullSubjectEvaluator)
    ev.calculator = _Stub()
    return ev


# --- v2 schema ---------------------------------------------------------------

@pytest.mark.parametrize("v2_path", sorted(list(V2_EN_DIR.glob("*.csv")) +
                                         list(V2_ES_DIR.glob("*.csv"))))
def test_v2_loads_and_normalizes(v2_path):
    ev = _make_evaluator()
    df = ev.load_stimuli_file(str(v2_path))

    # Normalized v1 names present
    for col in ("item", "item_group", "c_english", "target", "hotspot_english",
                "pronoun_status"):
        assert col in df.columns, f"{v2_path.name}: missing normalized col {col}"

    # v2-only fields preserved
    assert "condition" in df.columns
    assert "language" in df.columns
    assert "hotspot_position" in df.columns
    assert (df["schema_version"] == "v2").all()


@pytest.mark.parametrize("v2_path", sorted(list(V2_EN_DIR.glob("*.csv")) +
                                         list(V2_ES_DIR.glob("*.csv"))))
def test_v2_hotspot_alignment(v2_path):
    """For every v2 row, target.split()[hotspot_position] must equal hotspot_token."""
    ev = _make_evaluator()
    df = ev.load_stimuli_file(str(v2_path))
    for _, row in df.iterrows():
        toks = str(row["target"]).split()
        pos = int(row["hotspot_position"])
        assert 0 <= pos < len(toks), (
            f"{v2_path.name} item={row['item']} ps={row['pronoun_status']}: "
            f"hotspot_position {pos} out of range for target of length {len(toks)}"
        )
        assert toks[pos] == row["hotspot_english"], (
            f"{v2_path.name} item={row['item']} ps={row['pronoun_status']}: "
            f"toks[{pos}]={toks[pos]!r} != hotspot_token={row['hotspot_english']!r}"
        )


def test_v2_rejects_mismatched_hotspot(tmp_path):
    """A v2 CSV with a deliberately-wrong hotspot_position must raise."""
    bad = pd.DataFrame([
        {"item_id": 1, "category": "subject_drop", "condition": "subj_3sg",
         "pronoun_status": 1, "context": "ana ganó el premio .",
         "target": "ella muestra orgullo .", "hotspot_token": "muestra",
         "hotspot_position": 99, "language": "es"},
    ])
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)
    ev = _make_evaluator()
    with pytest.raises(ValueError, match="hotspot"):
        ev.load_stimuli_file(str(path))


# --- v1 backwards compatibility ---------------------------------------------

@pytest.mark.skipif(not V1_DIR.exists(), reason="v1 stimuli dir absent")
@pytest.mark.parametrize("v1_path", sorted(V1_DIR.glob("*.csv"))[:3])
def test_v1_still_loads(v1_path):
    """Existing v1 CSVs must continue to parse."""
    ev = _make_evaluator()
    df = ev.load_stimuli_file(str(v1_path))
    assert "item" in df.columns
    assert "c_english" in df.columns
    assert "hotspot_english" in df.columns
    assert df["schema_version"].iloc[0] in ("v1", "v1_master")


def test_v1_master_format(tmp_path):
    """v1 master files (with 'form' column) should be recognized."""
    master = pd.DataFrame([
        {"item": 1, "item_group": "1a_3rdSG", "form": "default",
         "pronoun_status": 1, "c_english": "marta won .",
         "target": "she shows pride .", "hotspot_english": "shows"},
        {"item": 1, "item_group": "1a_3rdSG", "form": "default",
         "pronoun_status": 0, "c_english": "marta won .",
         "target": "shows pride .", "hotspot_english": "shows"},
    ])
    path = tmp_path / "master.csv"
    master.to_csv(path, index=False)
    ev = _make_evaluator()
    df = ev.load_stimuli_file(str(path))
    assert (df["schema_version"] == "v1_master").all()
    assert "form" in df.columns
