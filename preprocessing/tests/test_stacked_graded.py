"""Tests for the stacked graded ablation (pronoun-drop × intervention).

Real-parser equivalence tests (en_core_web_sm) mirror the edit-plan
equivalence harness: the stacked composition must reduce to its two
constituents in the degenerate cases.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import spacy

import preprocessing.ablations  # noqa: F401 — trigger registration
from preprocessing.ablations.remove_subject_pronouns_graded import (
    GradedSubjectPronounRemover,
)
from preprocessing.ablations.stacked_graded import StackedGradedAblation
from preprocessing.registry import AblationRegistry

nlp = None
try:  # pragma: no cover
    nlp = spacy.load("en_core_web_sm")
except Exception:  # noqa: BLE001
    pass

requires_model = pytest.mark.skipif(nlp is None,
                                    reason="en_core_web_sm not installed")

LINES = [
    "He said that she would bring the books tomorrow.",
    "I think they walked home after the game ended.",
    "You know we never really wanted those results.",
    "The children played quietly in the sunny garden.",
]


@pytest.fixture
def selection_dir(tmp_path):
    """Select the first PRON-nsubj token of each line as decile 0."""
    rows = {"line_idx": [], "token_i": [], "info_decile": [], "rand_decile": []}
    for idx, line in enumerate(LINES):
        doc = nlp(line)
        prons = [t.i for t in doc if t.pos_ == "PRON" and "subj" in t.dep_]
        for j, ti in enumerate(prons):
            rows["line_idx"].append(idx)
            rows["token_i"].append(ti)
            rows["info_decile"].append(0 if j == 0 else 5)
            rows["rand_decile"].append(5)
    pq.write_table(pa.table(rows), tmp_path / "toy.parquet")
    return tmp_path


def configured(selection_dir, intervention, k=10, arm="info"):
    r = StackedGradedAblation()
    r.configure({"selection_dir": str(selection_dir), "arm": arm, "k": k,
                 "intervention": intervention})
    return r


@requires_model
class TestComposition:
    def test_baseline_equals_pure_graded(self, selection_dir):
        stacked = configured(selection_dir, "baseline", k=100)
        pure = GradedSubjectPronounRemover()
        pure.configure({"selection_dir": str(selection_dir),
                        "arm": "info", "k": 100})
        for idx, line in enumerate(LINES):
            doc = nlp(line)
            stacked.set_line_context("toy", idx)
            pure.set_line_context("toy", idx)
            assert stacked(doc) == pure(doc), f"line {idx} diverged"

    def test_no_selected_pronouns_equals_intervention_alone(self, selection_dir):
        # k=10 selects only decile-0 instances; a line whose pronouns are
        # all decile-5 must be byte-identical to the intervention alone.
        stacked = configured(selection_dir, "lemmatize_verbs", k=10)
        lem, _ = AblationRegistry.get("lemmatize_verbs")
        doc = nlp(LINES[3])  # "The children played..." — no PRON-nsubj sel
        stacked.set_line_context("toy", 3)
        s_text, s_n = stacked(doc)
        l_text, _ = lem(nlp(LINES[3]))
        assert s_text == l_text

    def test_merged_edits_and_deletion(self, selection_dir):
        stacked = configured(selection_dir, "lemmatize_verbs", k=100)
        doc = nlp(LINES[0])  # "He said that she would bring..."
        stacked.set_line_context("toy", 0)
        text, n = stacked(doc)
        assert n >= 2                      # He + she deleted
        assert "He" not in text and "she" not in text
        assert "say" in text               # said -> say (lemmatized)

    def test_bad_intervention_rejected(self, selection_dir):
        with pytest.raises(ValueError, match="intervention"):
            configured(selection_dir, "enrich_verbs")

    def test_registered(self):
        fn, _ = AblationRegistry.get("pronoun_drop_stacked")
        assert isinstance(fn, StackedGradedAblation)
        assert hasattr(fn, "configure") and hasattr(fn, "set_line_context")

    def test_expletive_line_removal_dominates(self, selection_dir):
        stacked = configured(selection_dir, "remove_expletive_sentences",
                             k=100)
        stacked.reset_file_state()
        doc = nlp("There is a book on the table.")
        stacked.set_line_context("toy", 0)
        text, n = stacked(doc)
        assert text == "" and n == 1       # tier-1 existential removed
