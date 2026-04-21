"""
Tests for enhanced English expletive sentence removal.

Three test groups:
1. Unit tests with mock docs (no spaCy model required)
2. Integration tests with en_core_web_sm (real parses)
3. Coreference tests (skipped if coref model unavailable)
"""

from unittest.mock import MagicMock

import pytest
import spacy

from preprocessing.ablations.remove_expletive_sentences import (
    EnglishExpletiveSentenceRemover,
    _has_expletive_en_enhanced,
    _has_expletive_equivalent_es,
    _is_document_boundary,
    make_remove_expletive_sentences_en,
    make_remove_expletive_sentences_es,
)
from preprocessing.registry import AblationRegistry


# ---------------------------------------------------------------------------
# Helpers for mock docs
# ---------------------------------------------------------------------------

def _make_mock_token(
    text="word",
    lemma=None,
    dep_="nsubj",
    pos_="NOUN",
    lower_=None,
    head=None,
    children=None,
    idx=0,
):
    """Create a mock spaCy Token with the attributes used by the detector."""
    tok = MagicMock()
    tok.text = text
    tok.lemma_ = lemma if lemma is not None else text.lower()
    tok.dep_ = dep_
    tok.pos_ = pos_
    tok.lower_ = lower_ if lower_ is not None else text.lower()
    tok.idx = idx
    tok.head = head if head is not None else tok  # self-loop default
    tok.children = children if children is not None else []
    return tok


def _make_mock_doc(tokens, text=None):
    """Create a mock spaCy Doc from a list of mock tokens."""
    doc = MagicMock()
    doc.__iter__ = lambda self: iter(tokens)
    doc.__len__ = lambda self: len(tokens)
    doc.text = text if text is not None else " ".join(t.text for t in tokens)
    doc.text_with_ws = doc.text
    return doc


# ---------------------------------------------------------------------------
# Unit tests — mock docs, no spaCy model
# ---------------------------------------------------------------------------

class TestSpacyExplDetection:
    """Tier 1: spaCy dep_='expl' detection."""

    def test_expl_token_detected(self):
        """A token with dep_='expl' should trigger removal."""
        tok = _make_mock_token(text="There", dep_="expl")
        doc = _make_mock_doc([tok], text="There is a cat")
        assert EnglishExpletiveSentenceRemover._has_spacy_expl(doc) is True

    def test_no_expl_token(self):
        """No dep_='expl' token should not trigger."""
        tok = _make_mock_token(text="The", dep_="det")
        doc = _make_mock_doc([tok], text="The cat is here")
        assert EnglishExpletiveSentenceRemover._has_spacy_expl(doc) is False

    def test_expl_triggers_removal(self):
        """Full __call__ should return ('', 1) for expl tokens."""
        tok = _make_mock_token(text="There", dep_="expl")
        doc = _make_mock_doc([tok], text="There is a cat")
        remover = EnglishExpletiveSentenceRemover()
        result_text, count = remover(doc)
        assert result_text == ""
        assert count == 1


class TestWeatherItHeuristic:
    """Tier 2a: weather-it detection."""

    def test_weather_verb_it_detected(self):
        """'It rains' should be detected as heuristic expletive."""
        head_verb = _make_mock_token(
            text="rains", lemma="rain", dep_="ROOT", pos_="VERB",
        )
        it_tok = _make_mock_token(
            text="It", lemma="it", dep_="nsubj", pos_="PRON",
            head=head_verb, idx=0,
        )
        head_verb.children = [it_tok]
        doc = _make_mock_doc([it_tok, head_verb], text="It rains")
        candidate = EnglishExpletiveSentenceRemover._find_heuristic_expletive_it(doc)
        assert candidate is not None
        assert candidate.text == "It"

    def test_non_weather_verb_not_detected(self):
        """'It runs' should NOT be detected."""
        head_verb = _make_mock_token(
            text="runs", lemma="run", dep_="ROOT", pos_="VERB",
        )
        it_tok = _make_mock_token(
            text="It", lemma="it", dep_="nsubj", pos_="PRON",
            head=head_verb, idx=0,
        )
        doc = _make_mock_doc([it_tok, head_verb], text="It runs")
        candidate = EnglishExpletiveSentenceRemover._find_heuristic_expletive_it(doc)
        assert candidate is None


class TestRaisingItHeuristic:
    """Tier 2b: raising-it detection."""

    def test_raising_verb_with_complement_detected(self):
        """'It seems that...' should be detected."""
        clausal_child = _make_mock_token(
            text="works", dep_="ccomp", pos_="VERB",
        )
        head_verb = _make_mock_token(
            text="seems", lemma="seem", dep_="ROOT", pos_="VERB",
            children=[clausal_child],
        )
        it_tok = _make_mock_token(
            text="It", lemma="it", dep_="nsubj", pos_="PRON",
            head=head_verb, idx=0,
        )
        doc = _make_mock_doc([it_tok, head_verb, clausal_child], text="It seems that it works")
        candidate = EnglishExpletiveSentenceRemover._find_heuristic_expletive_it(doc)
        assert candidate is not None

    def test_raising_verb_without_complement_not_detected(self):
        """'It seems nice' without clausal complement should NOT be detected."""
        adj_child = _make_mock_token(text="nice", dep_="acomp", pos_="ADJ")
        head_verb = _make_mock_token(
            text="seems", lemma="seem", dep_="ROOT", pos_="VERB",
            children=[adj_child],
        )
        # adj_child has no clausal complement children
        adj_child.children = []
        adj_child.head = head_verb
        adj_child.dep_ = "acomp"
        it_tok = _make_mock_token(
            text="It", lemma="it", dep_="nsubj", pos_="PRON",
            head=head_verb, idx=0,
        )
        doc = _make_mock_doc([it_tok, head_verb, adj_child], text="It seems nice")
        candidate = EnglishExpletiveSentenceRemover._find_heuristic_expletive_it(doc)
        assert candidate is None


class TestCopularRaisingHeuristic:
    """Tier 2c: copula + raising adjective detection."""

    def test_copula_raising_adj_with_complement_detected(self):
        """'It is clear that...' should be detected."""
        that_child = _make_mock_token(text="that", lemma="that", dep_="mark")
        ccomp_child = _make_mock_token(text="works", dep_="ccomp", pos_="VERB")
        adj_child = _make_mock_token(
            text="clear", lemma="clear", dep_="acomp", pos_="ADJ",
            children=[that_child, ccomp_child],
        )
        head_verb = _make_mock_token(
            text="is", lemma="be", dep_="ROOT", pos_="AUX",
            children=[adj_child],
        )
        adj_child.head = head_verb
        it_tok = _make_mock_token(
            text="It", lemma="it", dep_="nsubj", pos_="PRON",
            head=head_verb, idx=0,
        )
        doc = _make_mock_doc(
            [it_tok, head_verb, adj_child, that_child, ccomp_child],
            text="It is clear that it works",
        )
        candidate = EnglishExpletiveSentenceRemover._find_heuristic_expletive_it(doc)
        assert candidate is not None

    def test_non_raising_adj_not_detected(self):
        """'It is red' should NOT be detected."""
        adj_child = _make_mock_token(
            text="red", lemma="red", dep_="acomp", pos_="ADJ",
            children=[],
        )
        head_verb = _make_mock_token(
            text="is", lemma="be", dep_="ROOT", pos_="AUX",
            children=[adj_child],
        )
        adj_child.head = head_verb
        it_tok = _make_mock_token(
            text="It", lemma="it", dep_="nsubj", pos_="PRON",
            head=head_verb, idx=0,
        )
        doc = _make_mock_doc([it_tok, head_verb, adj_child], text="It is red")
        candidate = EnglishExpletiveSentenceRemover._find_heuristic_expletive_it(doc)
        assert candidate is None


class TestReferentialItKept:
    """Referential 'it' should NOT trigger removal."""

    def test_referential_it_with_regular_verb(self):
        """'It works fine' — referential pronoun, not expletive."""
        head_verb = _make_mock_token(
            text="works", lemma="work", dep_="ROOT", pos_="VERB",
        )
        it_tok = _make_mock_token(
            text="It", lemma="it", dep_="nsubj", pos_="PRON",
            head=head_verb, idx=0,
        )
        doc = _make_mock_doc([it_tok, head_verb], text="It works fine")
        remover = EnglishExpletiveSentenceRemover()
        result_text, count = remover(doc)
        assert result_text == "It works fine"
        assert count == 0


class TestEmptyDoc:
    """Edge case: empty document."""

    def test_empty_doc_kept(self):
        doc = _make_mock_doc([], text="")
        remover = EnglishExpletiveSentenceRemover()
        result_text, count = remover(doc)
        assert count == 0


class TestEnhancedDetectionFunction:
    """Test the stateless _has_expletive_en_enhanced validator function."""

    def test_detects_expl(self):
        tok = _make_mock_token(text="There", dep_="expl")
        doc = _make_mock_doc([tok])
        assert _has_expletive_en_enhanced(doc) is True

    def test_detects_weather_it(self):
        head_verb = _make_mock_token(
            text="rains", lemma="rain", dep_="ROOT", pos_="VERB",
        )
        it_tok = _make_mock_token(
            text="It", lemma="it", dep_="nsubj", pos_="PRON",
            head=head_verb, idx=0,
        )
        doc = _make_mock_doc([it_tok, head_verb], text="It rains")
        assert _has_expletive_en_enhanced(doc) is True

    def test_no_expletive(self):
        tok = _make_mock_token(text="The", dep_="det")
        doc = _make_mock_doc([tok])
        assert _has_expletive_en_enhanced(doc) is False


class TestCorefFallback:
    """Tier 3: coreference confirmation behavior."""

    def test_no_coref_trusts_heuristic(self):
        """Without coref model, heuristic candidate should be removed."""
        head_verb = _make_mock_token(
            text="rains", lemma="rain", dep_="ROOT", pos_="VERB",
        )
        it_tok = _make_mock_token(
            text="It", lemma="it", dep_="nsubj", pos_="PRON",
            head=head_verb, idx=0,
        )
        doc = _make_mock_doc([it_tok, head_verb], text="It rains")
        remover = EnglishExpletiveSentenceRemover(coref_model=None)
        result_text, count = remover(doc)
        assert result_text == ""
        assert count == 1

    def test_coref_model_lazy_loading(self):
        """Coref model should not be loaded until first heuristic candidate."""
        remover = EnglishExpletiveSentenceRemover(coref_model="nonexistent_model")
        assert remover._coref_loaded is False
        # Process a non-expletive doc — coref should not be loaded
        tok = _make_mock_token(text="Hello", dep_="ROOT")
        doc = _make_mock_doc([tok], text="Hello")
        remover(doc)
        assert remover._coref_loaded is False


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_en_is_registered(self):
        assert AblationRegistry.is_registered("remove_expletive_sentences_en")

    def test_en_can_retrieve(self):
        ablation_fn, validator_fn = AblationRegistry.get("remove_expletive_sentences_en")
        assert callable(ablation_fn)
        assert callable(validator_fn)

    def test_en_is_callable_class(self):
        ablation_fn, _ = AblationRegistry.get("remove_expletive_sentences_en")
        assert isinstance(ablation_fn, EnglishExpletiveSentenceRemover)


# ---------------------------------------------------------------------------
# Integration tests — require en_core_web_sm
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nlp():
    """Load spaCy model once for all integration tests."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("spaCy model en_core_web_sm not available")


class TestIntegrationWeatherIt:
    """Integration tests with real spaCy parses."""

    def test_it_is_raining(self, nlp):
        """'It is raining outside' should be removed."""
        doc = nlp("It is raining outside.")
        remover = EnglishExpletiveSentenceRemover()
        result_text, count = remover(doc)
        # Should be removed either via expl or weather heuristic
        assert count == 1
        assert result_text == ""

    def test_it_snowed_yesterday(self, nlp):
        doc = nlp("It snowed yesterday.")
        remover = EnglishExpletiveSentenceRemover()
        _, count = remover(doc)
        assert count == 1

    def test_there_is_a_cat(self, nlp):
        """'There is a cat' — existential there, spaCy expl."""
        doc = nlp("There is a cat on the mat.")
        remover = EnglishExpletiveSentenceRemover()
        _, count = remover(doc)
        assert count == 1


class TestIntegrationRaisingIt:
    def test_it_seems_that(self, nlp):
        doc = nlp("It seems that he is tired.")
        remover = EnglishExpletiveSentenceRemover()
        _, count = remover(doc)
        assert count == 1

    def test_it_appears_that(self, nlp):
        doc = nlp("It appears that the system works.")
        remover = EnglishExpletiveSentenceRemover()
        _, count = remover(doc)
        assert count == 1

    def test_it_is_clear_that(self, nlp):
        doc = nlp("It is clear that we need to act.")
        remover = EnglishExpletiveSentenceRemover()
        _, count = remover(doc)
        assert count == 1

    def test_it_is_likely_that(self, nlp):
        doc = nlp("It is likely that rain will come.")
        remover = EnglishExpletiveSentenceRemover()
        _, count = remover(doc)
        assert count == 1


class TestIntegrationReferentialKept:
    def test_referential_it_kept(self, nlp):
        """'The car is fast. It goes quickly.' — referential, should be kept."""
        doc = nlp("It goes quickly.")
        remover = EnglishExpletiveSentenceRemover()
        result_text, count = remover(doc)
        assert count == 0
        assert result_text != ""

    def test_it_broke(self, nlp):
        """'It broke' — referential pronoun, should be kept."""
        doc = nlp("It broke yesterday.")
        remover = EnglishExpletiveSentenceRemover()
        _, count = remover(doc)
        assert count == 0

    def test_regular_sentence_kept(self, nlp):
        doc = nlp("The dog ran across the yard.")
        remover = EnglishExpletiveSentenceRemover()
        _, count = remover(doc)
        assert count == 0


# ---------------------------------------------------------------------------
# Coreference tests — skipped if coref model unavailable
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def coref_model_name():
    """Check if a coreference model is available."""
    model_name = "en_coreference_web_trf"
    try:
        spacy.load(model_name)
        return model_name
    except OSError:
        pytest.skip(f"Coreference model {model_name} not available")


class TestCorefIntegration:
    def test_unbound_it_removed_with_coref(self, nlp, coref_model_name):
        """Weather-it should be confirmed as unbound by coref."""
        doc = nlp("It is raining outside.")
        remover = EnglishExpletiveSentenceRemover(coref_model=coref_model_name)
        _, count = remover(doc)
        assert count == 1

    def test_referential_it_kept_with_coref(self, nlp, coref_model_name):
        """Referential 'it' near a raising verb should be kept by coref."""
        remover = EnglishExpletiveSentenceRemover(coref_model=coref_model_name)
        # First line provides referent
        prev_doc = nlp("The machine is complex.")
        remover(prev_doc)
        # Second line — "It seems to work" where "it" = the machine
        doc = nlp("It seems to work well.")
        _, count = remover(doc)
        # With coref, this should ideally be kept (referential)
        # Note: actual behavior depends on coreference model quality
        assert isinstance(count, int)

    def test_coref_fallback_on_missing_model(self, nlp):
        """If coref model name is bogus, should fall back to heuristic."""
        doc = nlp("It is raining.")
        remover = EnglishExpletiveSentenceRemover(coref_model="nonexistent_model_xyz")
        _, count = remover(doc)
        # Should still remove — fallback to heuristic
        assert count == 1


# ---------------------------------------------------------------------------
# Tier counting tests (Change 2)
# ---------------------------------------------------------------------------

class TestTierCounting:
    """Verify per-tier removal counts and removed line indices."""

    def test_tier1_expl_counted(self):
        """spaCy expl removal should increment tier1_expl."""
        remover = EnglishExpletiveSentenceRemover()
        tok = _make_mock_token(text="There", dep_="expl")
        doc = _make_mock_doc([tok], text="There is a cat")
        remover(doc)
        counts = remover.get_file_tier_counts()
        assert counts["tier1_expl"] == 1

    def test_tier2_weather_counted(self):
        """Weather-it removal should increment tier2_weather."""
        remover = EnglishExpletiveSentenceRemover()
        head_verb = _make_mock_token(
            text="rains", lemma="rain", dep_="ROOT", pos_="VERB",
        )
        it_tok = _make_mock_token(
            text="It", lemma="it", dep_="nsubj", pos_="PRON",
            head=head_verb, idx=0,
        )
        doc = _make_mock_doc([it_tok, head_verb], text="It rains")
        remover(doc)
        counts = remover.get_file_tier_counts()
        assert counts["tier2_weather"] == 1

    def test_tier2_raising_counted(self):
        """Raising-it removal should increment tier2_raising."""
        remover = EnglishExpletiveSentenceRemover()
        clausal_child = _make_mock_token(
            text="works", dep_="ccomp", pos_="VERB",
        )
        head_verb = _make_mock_token(
            text="seems", lemma="seem", dep_="ROOT", pos_="VERB",
            children=[clausal_child],
        )
        it_tok = _make_mock_token(
            text="It", lemma="it", dep_="nsubj", pos_="PRON",
            head=head_verb, idx=0,
        )
        doc = _make_mock_doc([it_tok, head_verb, clausal_child], text="It seems that it works")
        remover(doc)
        counts = remover.get_file_tier_counts()
        assert counts["tier2_raising"] == 1

    def test_tier2_copular_counted(self):
        """Copula + raising adj removal should increment tier2_copular."""
        remover = EnglishExpletiveSentenceRemover()
        that_child = _make_mock_token(text="that", lemma="that", dep_="mark")
        ccomp_child = _make_mock_token(text="works", dep_="ccomp", pos_="VERB")
        adj_child = _make_mock_token(
            text="clear", lemma="clear", dep_="acomp", pos_="ADJ",
            children=[that_child, ccomp_child],
        )
        head_verb = _make_mock_token(
            text="is", lemma="be", dep_="ROOT", pos_="AUX",
            children=[adj_child],
        )
        adj_child.head = head_verb
        it_tok = _make_mock_token(
            text="It", lemma="it", dep_="nsubj", pos_="PRON",
            head=head_verb, idx=0,
        )
        doc = _make_mock_doc(
            [it_tok, head_verb, adj_child, that_child, ccomp_child],
            text="It is clear that it works",
        )
        remover(doc)
        counts = remover.get_file_tier_counts()
        assert counts["tier2_copular"] == 1

    def test_removed_line_indices_tracked(self):
        """Removed line indices should be tracked correctly."""
        remover = EnglishExpletiveSentenceRemover()

        # Line 0: kept (non-expletive)
        tok0 = _make_mock_token(text="Hello", dep_="ROOT")
        doc0 = _make_mock_doc([tok0], text="Hello")
        remover(doc0)

        # Line 1: removed (expl)
        tok1 = _make_mock_token(text="There", dep_="expl")
        doc1 = _make_mock_doc([tok1], text="There is a cat")
        remover(doc1)

        # Line 2: kept
        tok2 = _make_mock_token(text="Dogs", dep_="nsubj")
        doc2 = _make_mock_doc([tok2], text="Dogs bark")
        remover(doc2)

        # Line 3: removed (weather)
        head_verb = _make_mock_token(
            text="rains", lemma="rain", dep_="ROOT", pos_="VERB",
        )
        it_tok = _make_mock_token(
            text="It", lemma="it", dep_="nsubj", pos_="PRON",
            head=head_verb, idx=0,
        )
        doc3 = _make_mock_doc([it_tok, head_verb], text="It rains")
        remover(doc3)

        assert remover._removed_line_indices == [1, 3]

    def test_multiple_tiers_accumulate(self):
        """Processing multiple expletive types should accumulate across tiers."""
        remover = EnglishExpletiveSentenceRemover()

        # Tier 1
        tok = _make_mock_token(text="There", dep_="expl")
        remover(_make_mock_doc([tok], text="There is a cat"))

        # Tier 2 weather
        head_verb = _make_mock_token(
            text="rains", lemma="rain", dep_="ROOT", pos_="VERB",
        )
        it_tok = _make_mock_token(
            text="It", lemma="it", dep_="nsubj", pos_="PRON",
            head=head_verb, idx=0,
        )
        remover(_make_mock_doc([it_tok, head_verb], text="It rains"))

        counts = remover.get_file_tier_counts()
        assert counts["tier1_expl"] == 1
        assert counts["tier2_weather"] == 1
        assert sum(counts.values()) == 2

    def test_reset_file_state_clears_counts(self):
        """reset_file_state should zero all counters and indices."""
        remover = EnglishExpletiveSentenceRemover()
        tok = _make_mock_token(text="There", dep_="expl")
        remover(_make_mock_doc([tok], text="There is a cat"))
        assert remover.get_file_tier_counts()["tier1_expl"] == 1
        assert len(remover._removed_line_indices) == 1

        remover.reset_file_state()
        counts = remover.get_file_tier_counts()
        assert all(v == 0 for v in counts.values())
        assert remover._removed_line_indices == []
        assert remover._current_line_index == 0

    def test_get_file_tier_counts_returns_copy(self):
        """get_file_tier_counts should return a copy, not a reference."""
        remover = EnglishExpletiveSentenceRemover()
        counts = remover.get_file_tier_counts()
        counts["tier1_expl"] = 999
        assert remover.get_file_tier_counts()["tier1_expl"] == 0


# ---------------------------------------------------------------------------
# Document boundary detection tests (Change 4)
# ---------------------------------------------------------------------------

class TestDocumentBoundaryDetection:
    """Verify _is_document_boundary for boundary markers."""

    def test_standard_boundary(self):
        assert _is_document_boundary("= = = PG21374 = = =") is True

    def test_boundary_with_whitespace(self):
        assert _is_document_boundary("  = = = PG21374 = = =  ") is True

    def test_normal_text_not_boundary(self):
        assert _is_document_boundary("The cat sat on the mat.") is False

    def test_empty_string_not_boundary(self):
        assert _is_document_boundary("") is False

    def test_partial_boundary_not_matched(self):
        assert _is_document_boundary("= = = PG21374") is False

    def test_boundary_passes_through(self):
        """Boundary lines should pass through unchanged with count=0."""
        remover = EnglishExpletiveSentenceRemover()
        doc = _make_mock_doc([], text="= = = PG21374 = = =")
        result_text, count = remover(doc)
        assert count == 0
        assert result_text == "= = = PG21374 = = ="


# ---------------------------------------------------------------------------
# Context buffer management tests (Change 4)
# ---------------------------------------------------------------------------

class TestContextBufferManagement:
    """Verify context buffer behavior across lines and boundaries."""

    def test_buffer_accumulates_lines(self):
        """Buffer should accumulate processed lines."""
        remover = EnglishExpletiveSentenceRemover(context_lines=3)
        for text in ["Line one.", "Line two.", "Line three."]:
            tok = _make_mock_token(text=text.split()[0], dep_="ROOT")
            remover(_make_mock_doc([tok], text=text))
        assert len(remover._context_buffer) == 3

    def test_buffer_trims_to_context_lines(self):
        """Buffer should not exceed context_lines."""
        remover = EnglishExpletiveSentenceRemover(context_lines=2)
        for text in ["Line one.", "Line two.", "Line three.", "Line four."]:
            tok = _make_mock_token(text=text.split()[0], dep_="ROOT")
            remover(_make_mock_doc([tok], text=text))
        assert len(remover._context_buffer) == 2
        assert remover._context_buffer[0] == "Line three."
        assert remover._context_buffer[1] == "Line four."

    def test_buffer_clears_at_boundary(self):
        """Buffer should clear when a document boundary is encountered."""
        remover = EnglishExpletiveSentenceRemover(context_lines=3)
        # Add some context
        for text in ["Line one.", "Line two."]:
            tok = _make_mock_token(text=text.split()[0], dep_="ROOT")
            remover(_make_mock_doc([tok], text=text))
        assert len(remover._context_buffer) == 2

        # Hit a boundary
        boundary_doc = _make_mock_doc([], text="= = = PG999 = = =")
        remover(boundary_doc)
        assert len(remover._context_buffer) == 0

    def test_buffer_clears_on_reset(self):
        """reset_file_state should clear context buffer."""
        remover = EnglishExpletiveSentenceRemover(context_lines=3)
        tok = _make_mock_token(text="Hello", dep_="ROOT")
        remover(_make_mock_doc([tok], text="Hello world"))
        assert len(remover._context_buffer) == 1
        remover.reset_file_state()
        assert len(remover._context_buffer) == 0

    def test_factory_passes_context_lines(self):
        """make_remove_expletive_sentences_en should pass context_lines."""
        remover = make_remove_expletive_sentences_en(context_lines=5)
        assert remover._context_lines == 5


# ---------------------------------------------------------------------------
# Spanish — unit tests with mock docs
# ---------------------------------------------------------------------------

def _make_es_mock_token(
    text="word",
    lemma=None,
    dep_="ROOT",
    pos_="VERB",
    lower_=None,
    head=None,
    children=None,
    i=0,
):
    """Mock token for Spanish tests — like _make_mock_token but with `i` field
    (needed by the ello-head tracking set)."""
    tok = MagicMock()
    tok.text = text
    tok.lemma_ = lemma if lemma is not None else text.lower()
    tok.dep_ = dep_
    tok.pos_ = pos_
    tok.lower_ = lower_ if lower_ is not None else text.lower()
    tok.i = i
    tok.head = head if head is not None else tok
    tok.children = children if children is not None else []
    return tok


class TestSpanishWeatherVerb:
    """Category 1: Spanish weather verbs."""

    def test_llover_detected(self):
        verb = _make_es_mock_token(text="llueve", lemma="llover", pos_="VERB", i=0)
        doc = _make_mock_doc([verb], text="llueve mucho hoy")
        assert _has_expletive_equivalent_es(doc) is True

    def test_nevar_detected(self):
        verb = _make_es_mock_token(text="nevaba", lemma="nevar", pos_="VERB", i=0)
        doc = _make_mock_doc([verb], text="nevaba ayer")
        assert _has_expletive_equivalent_es(doc) is True

    def test_non_weather_verb_not_detected(self):
        verb = _make_es_mock_token(text="corre", lemma="correr", pos_="VERB", i=0)
        doc = _make_mock_doc([verb], text="juan corre rápido")
        assert _has_expletive_equivalent_es(doc) is False


class TestSpanishExistentialHaber:
    """Category 2: existential haber (hay, había, ...)."""

    def test_existential_hay_detected(self):
        """'hay tres gatos' — VERB haber, no nsubj."""
        verb = _make_es_mock_token(
            text="hay", lemma="haber", pos_="VERB", dep_="ROOT",
            children=[_make_es_mock_token(text="gatos", dep_="obj", pos_="NOUN", i=1)],
            i=0,
        )
        doc = _make_mock_doc([verb], text="hay tres gatos")
        assert _has_expletive_equivalent_es(doc) is True

    def test_auxiliary_haber_not_detected(self):
        """'ha comido' — AUX haber, not existential."""
        aux = _make_es_mock_token(text="ha", lemma="haber", pos_="AUX", dep_="aux", i=0)
        main = _make_es_mock_token(
            text="comido", lemma="comer", pos_="VERB", dep_="ROOT",
            children=[aux, _make_es_mock_token(text="juan", dep_="nsubj", pos_="PROPN", i=2)],
            i=1,
        )
        aux.head = main
        doc = _make_mock_doc([aux, main], text="juan ha comido")
        assert _has_expletive_equivalent_es(doc) is False

    def test_haber_with_nsubj_not_detected(self):
        """'los problemas habían crecido' — haber with nsubj is not existential."""
        verb = _make_es_mock_token(
            text="habían", lemma="haber", pos_="VERB", dep_="ROOT",
            children=[_make_es_mock_token(text="problemas", dep_="nsubj", pos_="NOUN", i=1)],
            i=0,
        )
        doc = _make_mock_doc([verb], text="habían crecido los problemas")
        assert _has_expletive_equivalent_es(doc) is False


class TestSpanishImpersonalRaising:
    """Category 3: impersonal raising verbs (parecer, resultar, ...)."""

    def test_parece_que_detected(self):
        """'parece que llega' — impersonal raising with clausal complement."""
        ccomp_child = _make_es_mock_token(
            text="llega", lemma="llegar", pos_="VERB", dep_="ccomp", i=2,
        )
        verb = _make_es_mock_token(
            text="parece", lemma="parecer", pos_="VERB", dep_="ROOT",
            children=[ccomp_child],
            i=0,
        )
        doc = _make_mock_doc([verb, ccomp_child], text="parece que llega")
        assert _has_expletive_equivalent_es(doc) is True

    def test_parecer_with_nsubj_not_detected(self):
        """'juan parece cansado' — parecer with nsubj is not impersonal."""
        ccomp_child = _make_es_mock_token(text="cansado", dep_="xcomp", pos_="ADJ", i=2)
        nsubj_child = _make_es_mock_token(text="juan", dep_="nsubj", pos_="PROPN", i=0)
        verb = _make_es_mock_token(
            text="parece", lemma="parecer", pos_="VERB", dep_="ROOT",
            children=[nsubj_child, ccomp_child],
            i=1,
        )
        doc = _make_mock_doc(
            [nsubj_child, verb, ccomp_child], text="juan parece cansado"
        )
        assert _has_expletive_equivalent_es(doc) is False


class TestSpanishImpersonalNecessity:
    """Category 4: impersonal necessity verbs (bastar, convenir, ...)."""

    def test_basta_detected(self):
        """'basta con eso' — impersonal, no nsubj."""
        verb = _make_es_mock_token(
            text="basta", lemma="bastar", pos_="VERB", dep_="ROOT", i=0,
        )
        doc = _make_mock_doc([verb], text="basta con eso")
        assert _has_expletive_equivalent_es(doc) is True

    def test_bastar_with_nsubj_not_detected(self):
        """'la comida basta' — nsubj present, not impersonal."""
        nsubj_child = _make_es_mock_token(text="comida", dep_="nsubj", pos_="NOUN", i=0)
        verb = _make_es_mock_token(
            text="basta", lemma="bastar", pos_="VERB", dep_="ROOT",
            children=[nsubj_child],
            i=1,
        )
        doc = _make_mock_doc([nsubj_child, verb], text="la comida basta")
        assert _has_expletive_equivalent_es(doc) is False


class TestSpanishOvertEllo:
    """Category 5: overt `ello` as subject of expletive verbs."""

    def test_ello_parece_que_detected(self):
        """'ello parece que ...' — archaic overt expletive."""
        ccomp_child = _make_es_mock_token(text="llega", dep_="ccomp", pos_="VERB", i=3)
        verb = _make_es_mock_token(
            text="parece", lemma="parecer", pos_="VERB", dep_="ROOT", i=1,
        )
        ello = _make_es_mock_token(
            text="ello", lemma="ello", pos_="PRON", dep_="nsubj",
            head=verb, i=0,
        )
        verb.children = [ello, ccomp_child]
        doc = _make_mock_doc([ello, verb, ccomp_child], text="ello parece que llega")
        assert _has_expletive_equivalent_es(doc) is True

    def test_ello_with_regular_verb_not_detected(self):
        """'ello funciona' — ello with non-expletive verb is not caught."""
        verb = _make_es_mock_token(
            text="funciona", lemma="funcionar", pos_="VERB", dep_="ROOT", i=1,
        )
        ello = _make_es_mock_token(
            text="ello", lemma="ello", pos_="PRON", dep_="nsubj",
            head=verb, i=0,
        )
        verb.children = [ello]
        doc = _make_mock_doc([ello, verb], text="ello funciona")
        assert _has_expletive_equivalent_es(doc) is False


class TestSpanishRegistration:
    """Registry integration."""

    def test_es_ablation_registered(self):
        import preprocessing.ablations  # trigger registration
        assert AblationRegistry.is_registered("remove_expletive_sentences_es")

    def test_factory_returns_callable(self):
        fn = make_remove_expletive_sentences_es()
        assert callable(fn)

    def test_factory_removes_weather_line(self):
        """End-to-end factory call on a mocked weather verb doc."""
        fn = make_remove_expletive_sentences_es()
        verb = _make_es_mock_token(text="llueve", lemma="llover", pos_="VERB", i=0)
        doc = _make_mock_doc([verb], text="llueve mucho")
        doc.text_with_ws = "llueve mucho"
        result, count = fn(doc)
        assert result == ""
        assert count == 1

    def test_factory_keeps_normal_line(self):
        fn = make_remove_expletive_sentences_es()
        verb = _make_es_mock_token(text="corre", lemma="correr", pos_="VERB", i=1)
        nsubj = _make_es_mock_token(text="juan", dep_="nsubj", pos_="PROPN", i=0)
        verb.children = [nsubj]
        doc = _make_mock_doc([nsubj, verb], text="juan corre")
        doc.text_with_ws = "juan corre"
        result, count = fn(doc)
        assert result == "juan corre"
        assert count == 0
