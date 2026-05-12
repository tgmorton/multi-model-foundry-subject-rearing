"""Tests for enrich_verbal_morphology, including the past-tense paradigm."""

import pytest
import spacy

from preprocessing.ablations.enrich_verbal_morphology import (
    enrich_verbal_morphology_doc,
    DEFAULT_SUFFIX_MAP,
    DEFAULT_PAST_SUFFIX_MAP,
)


@pytest.fixture(scope="module")
def nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("spaCy en_core_web_sm not available")


def _ablate(nlp, text):
    return enrich_verbal_morphology_doc(nlp(text))


def test_paradigms_are_disjoint():
    """Past and present paradigms must not share suffixes — otherwise
    the past/present distinction would not be recoverable on the surface."""
    pres = set(DEFAULT_SUFFIX_MAP.values())
    past = set(DEFAULT_PAST_SUFFIX_MAP.values())
    assert pres.isdisjoint(past), (
        f"present + past paradigms overlap: {pres & past}"
    )


def test_present_3sg_gets_present_suffix(nlp):
    out, n = _ablate(nlp, "She walks home.")
    assert "walkat" in out, f"expected walkat in {out!r}"
    assert n >= 1


def test_present_3pl_gets_present_suffix(nlp):
    out, n = _ablate(nlp, "They walk home.")
    assert "walkant" in out, f"expected walkant in {out!r}"
    assert n >= 1


def test_past_3sg_gets_past_suffix(nlp):
    out, n = _ablate(nlp, "She walked home.")
    assert "walkit" in out, f"expected walkit (past 3sg) in {out!r}"
    assert n >= 1


def test_past_3pl_gets_past_suffix(nlp):
    out, n = _ablate(nlp, "They walked home.")
    assert "walkerunt" in out, f"expected walkerunt (past 3pl) in {out!r}"
    assert n >= 1


def test_past_1sg_gets_past_suffix(nlp):
    out, n = _ablate(nlp, "I walked home.")
    assert "walki" in out, f"expected walki (past 1sg) in {out!r}"
    assert n >= 1


def test_participle_untouched(nlp):
    # "given" is a past participle (Form=Part), not finite past
    out, _ = _ablate(nlp, "She has given the book.")
    # Past participle should not get a past or present suffix
    assert "givenit" not in out and "givenerunt" not in out and "givenat" not in out


def test_irregular_past_uses_lemma_stem(nlp):
    """The past-tense 'ran' should be lemmatized to 'run' before the
    suffix is appended — surface should contain 'runit', not 'ranit'."""
    out, _ = _ablate(nlp, "She ran fast.")
    assert "runit" in out, f"expected runit (lemma+past3sg suffix) in {out!r}"
