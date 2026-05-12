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
    assert "walkakt" in out, f"expected walkakt in {out!r}"
    assert n >= 1


def test_present_2sg_gets_present_suffix(nlp):
    """Specifically verify -aks (2sg pres) is used, not the colliding -as."""
    out, n = _ablate(nlp, "You walk home.")
    assert "walkaks" in out, f"expected walkaks in {out!r}"
    assert " walk as " not in " " + out + " "  # no English-collision artefact
    assert n >= 1


def test_present_3pl_gets_present_suffix(nlp):
    out, n = _ablate(nlp, "They walk home.")
    assert "walkant" in out, f"expected walkant in {out!r}"
    assert n >= 1


def test_past_3sg_gets_past_suffix(nlp):
    """Specifically verify -ikt (3sg past) is used, not the colliding -it."""
    out, n = _ablate(nlp, "She walked home.")
    assert "walkikt" in out, f"expected walkikt (past 3sg) in {out!r}"
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
    assert "givenikt" not in out and "givenerunt" not in out and "givenakt" not in out


def test_irregular_past_uses_lemma_stem(nlp):
    """The past-tense 'ran' should be lemmatized to 'run' before the
    suffix is appended — surface should contain 'runikt', not 'ranikt'."""
    out, _ = _ablate(nlp, "She ran fast.")
    assert "runikt" in out, f"expected runikt (lemma+past3sg suffix) in {out!r}"


def test_no_suffix_is_english_homograph():
    """Sanity: walk every value in the present and past suffix maps and
    confirm none is identical to a high-frequency English function word."""
    forbidden = {"as", "at", "it", "is", "in", "on", "of", "or", "to", "by",
                 "be", "do", "an", "us", "we", "me", "my"}
    for v in DEFAULT_SUFFIX_MAP.values():
        assert v not in forbidden, (
            f"present suffix {v!r} collides with English function word"
        )
    for v in DEFAULT_PAST_SUFFIX_MAP.values():
        assert v not in forbidden, (
            f"past suffix {v!r} collides with English function word"
        )


# ---------------------------------------------------------------------------
# Suppletive (irregular) paradigm tests for be / have / do / go
# ---------------------------------------------------------------------------

def test_be_3sg_pres_uses_suppletive_est_not_beat(nlp):
    """The regular rule would give 'beat' (be + at), which collides with
    the English past tense of 'to beat'. Suppletion should produce 'est'."""
    out, _ = _ablate(nlp, "She is happy.")
    assert "est" in out.split(), f"expected 'est' (suppletive 3sg pres of be) in {out!r}"
    assert "beat" not in out.split(), f"surface 'beat' is the collision we wanted to avoid: {out!r}"


def test_be_past_uses_suppletive_fuit(nlp):
    out, _ = _ablate(nlp, "She was happy.")
    assert "fuit" in out.split(), f"expected 'fuit' (suppletive 3sg past of be) in {out!r}"


def test_go_3sg_pres_uses_suppletive_vadat_not_goat(nlp):
    """Regular rule would give 'goat' (go + at). Suppletion: 'vadat'."""
    out, _ = _ablate(nlp, "She goes home.")
    assert "vadat" in out.split(), f"expected 'vadat' in {out!r}"
    assert "goat" not in out.split(), f"surface 'goat' is the collision: {out!r}"


def test_have_3sg_pres_uses_suppletive_habat(nlp):
    out, _ = _ablate(nlp, "She has a book.")
    assert "habat" in out.split(), f"expected 'habat' (suppletive 3sg pres of have) in {out!r}"


def test_do_3sg_pres_uses_suppletive_facit(nlp):
    out, _ = _ablate(nlp, "She does it.")
    assert "facit" in out.split(), f"expected 'facit' (suppletive 3sg pres of do) in {out!r}"


def test_regular_verbs_still_use_suffix_rule(nlp):
    """The suppletion should ONLY apply to be/have/do/go. Other verbs
    must still follow the regular lemma+suffix rule."""
    out, _ = _ablate(nlp, "She walks home.")
    assert "walkat" in out, f"expected walkat (regular) in {out!r}"


def test_coarse_3sg_fallback_for_subjectless_finite_verb(nlp):
    """When the parse can't identify a subject for a finite verb, the
    rule should still apply a suffix using the 3sg default — otherwise
    we silently regress to bare-lemma output identical to lemmatize_verbs."""
    # Dialogue-tag inversion ("said Lucas") is a common case where
    # spaCy fails to link the verb to its subject via nsubj. We pick a
    # short sentence that produces this construction.
    out, n = _ablate(nlp, '"OK," said Lucas.')
    # `said` should be lemmatized to `say` and given the past 3sg suffix
    # (-it). Without the fallback, the output would just be `say`.
    assert "sayit" in out, f"expected 'sayit' (past 3sg fallback) in {out!r}"
    assert n >= 1


def test_fallback_disabled_when_default_is_none(nlp):
    """Passing default_person_number=None restores the legacy behaviour
    where unresolvable-subject verbs fall through to bare lemma."""
    from preprocessing.ablations.enrich_verbal_morphology import (
        _enrich_verbal_morphology, DEFAULT_SUFFIX_MAP, DEFAULT_PAST_SUFFIX_MAP,
        IRREGULAR_PARADIGMS,
    )
    doc = nlp('"OK," said Lucas.')
    out, _ = _enrich_verbal_morphology(
        doc, DEFAULT_SUFFIX_MAP, DEFAULT_PAST_SUFFIX_MAP, IRREGULAR_PARADIGMS,
        default_person_number=None,
    )
    # With the fallback disabled, `said` falls through to bare lemma `say`
    # with no suffix attached.
    assert " say " in out or out.endswith("say."), f"expected bare 'say' in {out!r}"
    assert "sayit" not in out


def test_no_suppletive_form_collides_with_english_homograph():
    """Sanity: walk over every suppletive form and confirm none is
    accidentally identical to a known high-frequency English word."""
    from preprocessing.ablations.enrich_verbal_morphology import IRREGULAR_PARADIGMS
    forbidden = {"beat", "goat", "goo", "habit", "facet"}
    all_forms = []
    for lemma, tenses in IRREGULAR_PARADIGMS.items():
        for tense, paradigm in tenses.items():
            for (pers, num), form in paradigm.items():
                all_forms.append((lemma, tense, pers, num, form))
                assert form not in forbidden, (
                    f"{lemma}/{tense}/{pers}{num} = {form!r} collides with a real English word"
                )
    # 4 verbs × 2 tenses × 6 forms = 48
    assert len(all_forms) == 48
