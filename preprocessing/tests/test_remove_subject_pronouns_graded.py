"""Tests for the graded subject-pronoun removal ablation (step 3)."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import spacy
from spacy.tokens import Doc

from preprocessing.ablations.remove_subject_pronouns_graded import (
    GradedSubjectPronounRemover,
    validate_graded_removal,
)


@pytest.fixture
def vocab():
    return spacy.blank("en").vocab


def make_doc(vocab, words, spaces):
    return Doc(vocab, words=words, spaces=spaces)


@pytest.fixture
def selection_dir(tmp_path):
    # switchboard.parquet: three instances across two lines with a spread
    # of deciles; rand deciles deliberately differ from info deciles.
    tbl = pa.table({
        "line_idx": [0, 0, 2],
        "token_i": [0, 3, 1],
        "info_decile": [0, 2, 5],
        "rand_decile": [5, 0, 0],
    })
    pq.write_table(tbl, tmp_path / "switchboard.parquet")
    return tmp_path


def configured(selection_dir, arm="info", k=10):
    r = GradedSubjectPronounRemover()
    r.configure({"selection_dir": str(selection_dir), "arm": arm, "k": k})
    return r


class TestConfigure:
    def test_missing_params(self, selection_dir):
        r = GradedSubjectPronounRemover()
        with pytest.raises(ValueError, match="needs parameters"):
            r.configure({"arm": "info"})

    def test_bad_arm(self, selection_dir):
        r = GradedSubjectPronounRemover()
        with pytest.raises(ValueError, match="arm"):
            r.configure({"selection_dir": str(selection_dir),
                         "arm": "informed", "k": 10})

    def test_bad_k(self, selection_dir):
        r = GradedSubjectPronounRemover()
        with pytest.raises(ValueError, match="k must be"):
            r.configure({"selection_dir": str(selection_dir),
                         "arm": "info", "k": 15})

    def test_missing_dir(self, tmp_path):
        r = GradedSubjectPronounRemover()
        with pytest.raises(FileNotFoundError):
            r.configure({"selection_dir": str(tmp_path / "nope"),
                         "arm": "info", "k": 10})


class TestRemoval:
    def test_requires_line_context(self, vocab, selection_dir):
        r = configured(selection_dir)
        doc = make_doc(vocab, ["He", "ran"], [True, False])
        with pytest.raises(RuntimeError, match="annotated-cache"):
            r(doc)

    def test_context_consumed_once(self, vocab, selection_dir):
        r = configured(selection_dir)
        doc = make_doc(vocab, ["He", "ran"], [True, False])
        r.set_line_context("switchboard", 1)
        r(doc)
        with pytest.raises(RuntimeError):
            r(doc)

    def test_k10_removes_decile0_only(self, vocab, selection_dir):
        r = configured(selection_dir, arm="info", k=10)
        # line 0: token 0 (decile 0) removed; token 3 (decile 2) kept
        doc = make_doc(vocab, ["He", "said", "that", "she", "ran"],
                       [True, True, True, True, False])
        r.set_line_context("switchboard", 0)
        text, n = r(doc)
        assert text == "said that she ran"
        assert n == 1

    def test_k30_removes_decile0_to_2(self, vocab, selection_dir):
        r = configured(selection_dir, arm="info", k=30)
        doc = make_doc(vocab, ["He", "said", "that", "she", "ran"],
                       [True, True, True, True, False])
        r.set_line_context("switchboard", 0)
        text, n = r(doc)
        assert text == "said that ran"
        assert n == 2

    def test_rand_arm_uses_rand_deciles(self, vocab, selection_dir):
        r = configured(selection_dir, arm="rand", k=10)
        doc = make_doc(vocab, ["He", "said", "that", "she", "ran"],
                       [True, True, True, True, False])
        r.set_line_context("switchboard", 0)
        text, n = r(doc)
        # rand_decile 0 is token 3 ("she"), not token 0
        assert text == "He said that ran"
        assert n == 1

    def test_untouched_line_returns_exact_text(self, vocab, selection_dir):
        r = configured(selection_dir, k=100)
        doc = make_doc(vocab, ["Nothing", "here"], [True, False])
        r.set_line_context("switchboard", 7)
        text, n = r(doc)
        assert text == doc.text
        assert n == 0

    def test_contraction_clitic_survives(self, vocab, selection_dir):
        # "He's happy" tokenized He + 's + happy; removing "He" leaves
        # "'s happy" (locked decision 2026-08-15: keep the contraction).
        tbl = pa.table({"line_idx": [4], "token_i": [0],
                        "info_decile": [0], "rand_decile": [0]})
        pq.write_table(tbl, selection_dir / "gutenberg.parquet")
        r = configured(selection_dir, k=10)
        doc = make_doc(vocab, ["He", "'s", "happy"], [False, True, False])
        assert doc.text == "He's happy"
        r.set_line_context("gutenberg", 4)
        text, n = r(doc)
        assert text == "'s happy"
        assert n == 1

    def test_missing_stem_raises(self, vocab, selection_dir):
        r = configured(selection_dir)
        with pytest.raises(FileNotFoundError, match="no selection table"):
            r.set_line_context("childes", 0)


class TestValidator:
    def test_accepts_shrink_and_equal(self):
        assert validate_graded_removal("He ran", "ran", None)
        assert validate_graded_removal("nothing", "nothing", None)
        assert not validate_graded_removal("a", "a b c", None)


class TestRegistration:
    def test_registered_with_configure_protocol(self):
        from preprocessing.registry import AblationRegistry
        fn, val = AblationRegistry.get("remove_subject_pronouns_graded")
        assert hasattr(fn, "configure")
        assert hasattr(fn, "set_line_context")
        assert val is validate_graded_removal
