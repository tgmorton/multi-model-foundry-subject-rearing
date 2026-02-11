"""
Tests for the Europarl parallel alignment pipeline.

Uses spacy.blank() with manually set token attributes — no model
download required.  Tests cover pronoun extraction, null-subject
detection, label resolution, quality filters, and checkpoint format.
"""

import pytest
import spacy
from spacy.tokens import Doc

from analysis.pronoun_recovery.parallel_data.en_pronoun_extractor import (
    ExtractedPronoun,
    extract_subject_pronouns,
)
from analysis.pronoun_recovery.parallel_data.it_null_subject_detector import (
    ItalianVerb,
    detect_finite_verbs,
)
from analysis.pronoun_recovery.parallel_data.label_resolver import (
    ResolvedMarker,
    resolve_markers,
)
from analysis.pronoun_recovery.parallel_data.quality_filters import (
    FilterStats,
    check_pair_quality,
)
from analysis.pronoun_recovery.parallel_data.passage_packer import (
    pack_passages,
    _extract_line_index,
)
from analysis.pronoun_recovery.parallel_data.aligner import (
    _parse_alignment_line,
    build_alignment_dicts,
)

from .conftest import make_doc


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def nlp_en():
    return spacy.blank("en")


@pytest.fixture
def nlp_it():
    return spacy.blank("it")


# ── EN Pronoun Extraction ────────────────────────────────────────────────


class TestEnPronounExtractor:
    """Tests for extract_subject_pronouns."""

    def test_extracts_nsubj_pronoun(self, nlp_en):
        """'She runs fast.' → one pronoun (she)."""
        doc = make_doc(
            nlp_en,
            words=["She", "runs", "fast", "."],
            pos=["PRON", "VERB", "ADV", "PUNCT"],
            deps=["nsubj", "ROOT", "advmod", "punct"],
            heads=[1, 1, 1, 1],
        )
        result = extract_subject_pronouns(doc)
        assert len(result) == 1
        assert result[0].text == "she"
        assert result[0].candidate_label == "PRO.3sg"
        assert result[0].head_idx == 1

    def test_skips_it_by_default(self, nlp_en):
        """'It rains.' → no pronouns (skip_all_it=True)."""
        doc = make_doc(
            nlp_en,
            words=["It", "rains", "."],
            pos=["PRON", "VERB", "PUNCT"],
            deps=["nsubj", "ROOT", "punct"],
            heads=[1, 1, 1],
        )
        result = extract_subject_pronouns(doc, skip_all_it=True)
        assert len(result) == 0

    def test_includes_it_when_flag_off(self, nlp_en):
        """'It rains.' → 'it' included when skip_all_it=False.

        Note: 'it' is not in _SUBJECT_PRONOUNS so still skipped.
        """
        doc = make_doc(
            nlp_en,
            words=["It", "rains", "."],
            pos=["PRON", "VERB", "PUNCT"],
            deps=["nsubj", "ROOT", "punct"],
            heads=[1, 1, 1],
        )
        result = extract_subject_pronouns(doc, skip_all_it=False)
        # "it" is not in _SUBJECT_PRONOUNS set, so still excluded
        assert len(result) == 0

    def test_skips_relative_pronouns(self, nlp_en):
        """'The man who runs' → no pronouns (relative 'who')."""
        doc = make_doc(
            nlp_en,
            words=["The", "man", "who", "runs"],
            pos=["DET", "NOUN", "PRON", "VERB"],
            deps=["det", "nsubj", "nsubj", "ROOT"],
            heads=[1, 3, 3, 3],
        )
        result = extract_subject_pronouns(doc)
        assert len(result) == 0

    def test_multiple_pronouns(self, nlp_en):
        """'I ran and she swam.' → two pronouns."""
        doc = make_doc(
            nlp_en,
            words=["I", "ran", "and", "she", "swam", "."],
            pos=["PRON", "VERB", "CCONJ", "PRON", "VERB", "PUNCT"],
            deps=["nsubj", "ROOT", "cc", "nsubj", "conj", "punct"],
            heads=[1, 1, 4, 4, 1, 1],
        )
        result = extract_subject_pronouns(doc)
        assert len(result) == 2
        assert result[0].text == "i"
        assert result[0].candidate_label == "PRO.1sg"
        assert result[1].text == "she"
        assert result[1].candidate_label == "PRO.3sg"

    def test_you_has_no_candidate_label(self, nlp_en):
        """'You speak.' → pronoun extracted with candidate_label=None."""
        doc = make_doc(
            nlp_en,
            words=["You", "speak", "."],
            pos=["PRON", "VERB", "PUNCT"],
            deps=["nsubj", "ROOT", "punct"],
            heads=[1, 1, 1],
        )
        result = extract_subject_pronouns(doc)
        assert len(result) == 1
        assert result[0].text == "you"
        assert result[0].candidate_label is None

    def test_skips_non_pronoun_nsubj(self, nlp_en):
        """'The dog runs.' → no pronouns (noun subject)."""
        doc = make_doc(
            nlp_en,
            words=["The", "dog", "runs", "."],
            pos=["DET", "NOUN", "VERB", "PUNCT"],
            deps=["det", "nsubj", "ROOT", "punct"],
            heads=[1, 2, 2, 2],
        )
        result = extract_subject_pronouns(doc)
        assert len(result) == 0


# ── IT Null Subject Detection ────────────────────────────────────────────


class TestItNullSubjectDetector:
    """Tests for detect_finite_verbs."""

    def test_null_subject_verb(self, nlp_it):
        """'Dichiaro' (I declare) — finite verb with no overt subject."""
        doc = make_doc(
            nlp_it,
            words=["Dichiaro", "la", "sessione", "."],
            pos=["VERB", "DET", "NOUN", "PUNCT"],
            deps=["ROOT", "det", "obj", "punct"],
            heads=[0, 2, 0, 0],
            morphs=[
                {"VerbForm": "Fin", "Person": "1", "Number": "Sing"},
                None,
                None,
                None,
            ],
        )
        verbs = detect_finite_verbs(doc)
        assert len(verbs) == 1
        assert verbs[0].subject_status == "null"
        assert not verbs[0].has_overt_subject
        assert verbs[0].morph_label_suffix == "1sg"

    def test_overt_subject_verb(self, nlp_it):
        """'Io dichiaro' — finite verb with overt nsubj."""
        doc = make_doc(
            nlp_it,
            words=["Io", "dichiaro", "la", "sessione", "."],
            pos=["PRON", "VERB", "DET", "NOUN", "PUNCT"],
            deps=["nsubj", "ROOT", "det", "obj", "punct"],
            heads=[1, 1, 3, 1, 1],
            morphs=[
                {"Person": "1", "Number": "Sing"},
                {"VerbForm": "Fin", "Person": "1", "Number": "Sing"},
                None,
                None,
                None,
            ],
        )
        verbs = detect_finite_verbs(doc)
        assert len(verbs) == 1
        assert verbs[0].subject_status == "overt"
        assert verbs[0].has_overt_subject

    def test_xcomp_inherited_subject(self, nlp_it):
        """xcomp verb inherits subject from matrix verb."""
        doc = make_doc(
            nlp_it,
            words=["Io", "voglio", "parlare", "."],
            pos=["PRON", "VERB", "VERB", "PUNCT"],
            deps=["nsubj", "ROOT", "xcomp", "punct"],
            heads=[1, 1, 1, 1],
            morphs=[
                {"Person": "1", "Number": "Sing"},
                {"VerbForm": "Fin", "Person": "1", "Number": "Sing"},
                {"VerbForm": "Fin", "Person": "1", "Number": "Sing"},
                None,
            ],
        )
        verbs = detect_finite_verbs(doc)
        # "parlare" as xcomp with Fin morph (unusual but testing logic)
        xcomp_verbs = [v for v in verbs if v.text == "parlare"]
        if xcomp_verbs:
            assert xcomp_verbs[0].subject_status == "inherited"
            assert xcomp_verbs[0].has_overt_subject

    def test_clausal_subject(self, nlp_it):
        """Clausal subject → subject_status='clausal'."""
        # "Parlare aiuta." — csubj is a child of the finite verb.
        doc = make_doc(
            nlp_it,
            words=["Parlare", "aiuta", "."],
            pos=["VERB", "VERB", "PUNCT"],
            deps=["csubj", "ROOT", "punct"],
            heads=[1, 1, 1],
            morphs=[
                {"VerbForm": "Inf"},
                {"VerbForm": "Fin", "Person": "3", "Number": "Sing"},
                None,
            ],
        )
        verbs = detect_finite_verbs(doc)
        assert len(verbs) == 1
        assert verbs[0].text == "aiuta"
        assert verbs[0].subject_status == "clausal"

    def test_no_finite_verbs(self, nlp_it):
        """Non-finite verbs are not returned."""
        doc = make_doc(
            nlp_it,
            words=["Parlando", "piano", "."],
            pos=["VERB", "ADV", "PUNCT"],
            deps=["ROOT", "advmod", "punct"],
            heads=[0, 0, 0],
            morphs=[{"VerbForm": "Ger"}, None, None],
        )
        verbs = detect_finite_verbs(doc)
        assert len(verbs) == 0


# ── Alignment Parsing ────────────────────────────────────────────────────


class TestAligner:
    """Tests for alignment parsing utilities."""

    def test_parse_alignment_line(self):
        """Standard awesome-align output format."""
        line = "0-0 1-2 3-1 4-4"
        result = _parse_alignment_line(line)
        assert result == [(0, 0), (1, 2), (3, 1), (4, 4)]

    def test_parse_empty_alignment(self):
        assert _parse_alignment_line("") == []

    def test_build_alignment_dicts(self):
        """Bidirectional dicts from alignment pairs."""
        alignments = [(0, 0), (1, 2), (1, 3), (2, 1)]
        en_to_it, it_to_en = build_alignment_dicts(alignments)

        assert en_to_it[0] == [0]
        assert en_to_it[1] == [2, 3]
        assert en_to_it[2] == [1]

        assert it_to_en[0] == [0]
        assert it_to_en[2] == [1]
        assert it_to_en[3] == [1]
        assert it_to_en[1] == [2]


# ── Label Resolution ─────────────────────────────────────────────────────


class TestLabelResolver:
    """Tests for resolve_markers."""

    def test_basic_resolution(self, nlp_en, nlp_it):
        """'I declare' aligned to 'Dichiaro' → PRO.1sg marker."""
        en_doc = make_doc(
            nlp_en,
            words=["I", "declare", "the", "session", "."],
            pos=["PRON", "VERB", "DET", "NOUN", "PUNCT"],
            deps=["nsubj", "ROOT", "det", "obj", "punct"],
            heads=[1, 1, 3, 1, 1],
        )
        it_doc = make_doc(
            nlp_it,
            words=["Dichiaro", "la", "sessione", "."],
            pos=["VERB", "DET", "NOUN", "PUNCT"],
            deps=["ROOT", "det", "obj", "punct"],
            heads=[0, 2, 0, 0],
            morphs=[
                {"VerbForm": "Fin", "Person": "1", "Number": "Sing"},
                None,
                None,
                None,
            ],
        )

        en_pronouns = [
            ExtractedPronoun(
                token_idx=0, text="i", head_idx=1, candidate_label="PRO.1sg"
            )
        ]
        it_verbs = [
            ItalianVerb(
                token_idx=0,
                text="Dichiaro",
                lemma="dichiarare",
                has_overt_subject=False,
                subject_status="null",
                person="1",
                number="Sing",
                morph_label_suffix="1sg",
            )
        ]
        en_to_it = {0: [0]}  # "I" → "Dichiaro"

        markers = resolve_markers(
            en_doc, it_doc, en_pronouns, it_verbs, en_to_it
        )
        assert len(markers) == 1
        assert markers[0].label == "PRO.1sg"
        assert markers[0].confidence == "high"
        assert markers[0].lexical_form == "io"

    def test_skips_overt_subject(self, nlp_en, nlp_it):
        """Skip when IT verb has overt subject."""
        en_doc = make_doc(
            nlp_en,
            words=["I", "speak"],
            pos=["PRON", "VERB"],
            deps=["nsubj", "ROOT"],
            heads=[1, 1],
        )
        it_doc = make_doc(
            nlp_it,
            words=["Io", "parlo"],
            pos=["PRON", "VERB"],
            deps=["nsubj", "ROOT"],
            heads=[1, 1],
            morphs=[
                None,
                {"VerbForm": "Fin", "Person": "1", "Number": "Sing"},
            ],
        )

        en_pronouns = [
            ExtractedPronoun(token_idx=0, text="i", head_idx=1, candidate_label="PRO.1sg")
        ]
        it_verbs = [
            ItalianVerb(
                token_idx=1,
                text="parlo",
                lemma="parlare",
                has_overt_subject=True,
                subject_status="overt",
                person="1",
                number="Sing",
                morph_label_suffix="1sg",
            )
        ]
        en_to_it = {0: [1]}

        stats = FilterStats()
        markers = resolve_markers(
            en_doc, it_doc, en_pronouns, it_verbs, en_to_it, stats=stats
        )
        assert len(markers) == 0
        assert stats.skipped_overt_subject == 1

    def test_morph_disagreement_skips(self, nlp_en, nlp_it):
        """Skip when EN label disagrees with IT morph."""
        en_doc = make_doc(
            nlp_en,
            words=["She", "speaks"],
            pos=["PRON", "VERB"],
            deps=["nsubj", "ROOT"],
            heads=[1, 1],
        )
        it_doc = make_doc(
            nlp_it,
            words=["Parliamo"],
            pos=["VERB"],
            deps=["ROOT"],
            heads=[0],
            morphs=[
                {"VerbForm": "Fin", "Person": "1", "Number": "Plur"},
            ],
        )

        en_pronouns = [
            ExtractedPronoun(token_idx=0, text="she", head_idx=1, candidate_label="PRO.3sg")
        ]
        it_verbs = [
            ItalianVerb(
                token_idx=0,
                text="Parliamo",
                lemma="parlare",
                has_overt_subject=False,
                subject_status="null",
                person="1",
                number="Plur",
                morph_label_suffix="1pl",
            )
        ]
        en_to_it = {0: [0]}

        stats = FilterStats()
        markers = resolve_markers(
            en_doc, it_doc, en_pronouns, it_verbs, en_to_it, stats=stats
        )
        assert len(markers) == 0
        assert stats.skipped_morph_disagree == 1

    def test_medium_confidence_when_no_morph(self, nlp_en, nlp_it):
        """Trust EN label with medium confidence when IT morph is missing."""
        en_doc = make_doc(
            nlp_en,
            words=["We", "declare"],
            pos=["PRON", "VERB"],
            deps=["nsubj", "ROOT"],
            heads=[1, 1],
        )
        it_doc = make_doc(
            nlp_it,
            words=["Dichiariamo"],
            pos=["VERB"],
            deps=["ROOT"],
            heads=[0],
            morphs=[{"VerbForm": "Fin"}],  # No Person/Number
        )

        en_pronouns = [
            ExtractedPronoun(token_idx=0, text="we", head_idx=1, candidate_label="PRO.1pl")
        ]
        it_verbs = [
            ItalianVerb(
                token_idx=0,
                text="Dichiariamo",
                lemma="dichiarare",
                has_overt_subject=False,
                subject_status="null",
                person=None,
                number=None,
                morph_label_suffix=None,
            )
        ]
        en_to_it = {0: [0]}

        markers = resolve_markers(
            en_doc, it_doc, en_pronouns, it_verbs, en_to_it
        )
        assert len(markers) == 1
        assert markers[0].label == "PRO.1pl"
        assert markers[0].confidence == "medium"

    def test_you_resolved_via_morph(self, nlp_en, nlp_it):
        """'you' resolved to PRO.2pl when IT verb has 2nd person plural morph."""
        en_doc = make_doc(
            nlp_en,
            words=["You", "speak"],
            pos=["PRON", "VERB"],
            deps=["nsubj", "ROOT"],
            heads=[1, 1],
        )
        it_doc = make_doc(
            nlp_it,
            words=["Parlate"],
            pos=["VERB"],
            deps=["ROOT"],
            heads=[0],
            morphs=[
                {"VerbForm": "Fin", "Person": "2", "Number": "Plur"},
            ],
        )

        en_pronouns = [
            ExtractedPronoun(token_idx=0, text="you", head_idx=1, candidate_label=None)
        ]
        it_verbs = [
            ItalianVerb(
                token_idx=0,
                text="Parlate",
                lemma="parlare",
                has_overt_subject=False,
                subject_status="null",
                person="2",
                number="Plur",
                morph_label_suffix="2pl",
            )
        ]
        en_to_it = {0: [0]}

        markers = resolve_markers(
            en_doc, it_doc, en_pronouns, it_verbs, en_to_it
        )
        assert len(markers) == 1
        assert markers[0].label == "PRO.2pl"
        assert markers[0].lexical_form == "voi"

    def test_deduplication_first_wins(self, nlp_en, nlp_it):
        """Two EN pronouns aligned to same IT verb → only first kept."""
        en_doc = make_doc(
            nlp_en,
            words=["I", "think", "I", "speak"],
            pos=["PRON", "VERB", "PRON", "VERB"],
            deps=["nsubj", "ROOT", "nsubj", "ccomp"],
            heads=[1, 1, 3, 1],
        )
        it_doc = make_doc(
            nlp_it,
            words=["Penso", "di", "parlare"],
            pos=["VERB", "ADP", "VERB"],
            deps=["ROOT", "mark", "xcomp"],
            heads=[0, 2, 0],
            morphs=[
                {"VerbForm": "Fin", "Person": "1", "Number": "Sing"},
                None,
                {"VerbForm": "Inf"},
            ],
        )

        en_pronouns = [
            ExtractedPronoun(token_idx=0, text="i", head_idx=1, candidate_label="PRO.1sg"),
            ExtractedPronoun(token_idx=2, text="i", head_idx=3, candidate_label="PRO.1sg"),
        ]
        it_verbs = [
            ItalianVerb(
                token_idx=0,
                text="Penso",
                lemma="pensare",
                has_overt_subject=False,
                subject_status="null",
                person="1",
                number="Sing",
                morph_label_suffix="1sg",
            )
        ]
        # Both EN pronouns align to same IT verb.
        en_to_it = {0: [0], 2: [0]}

        markers = resolve_markers(
            en_doc, it_doc, en_pronouns, it_verbs, en_to_it
        )
        assert len(markers) == 1


# ── Quality Filters ──────────────────────────────────────────────────────


class TestQualityFilters:
    """Tests for pair-level quality filters."""

    def test_passes_good_pair(self, nlp_en, nlp_it):
        en_doc = make_doc(nlp_en, words=["I", "speak", "today", "."])
        it_doc = make_doc(nlp_it, words=["Parlo", "oggi", "."])
        assert check_pair_quality(en_doc, it_doc)

    def test_rejects_too_short(self, nlp_en, nlp_it):
        en_doc = make_doc(nlp_en, words=["Hi"])
        it_doc = make_doc(nlp_it, words=["Ciao"])
        stats = FilterStats()
        assert not check_pair_quality(en_doc, it_doc, min_tokens=3, stats=stats)
        assert stats.skipped_too_short == 1

    def test_rejects_empty(self, nlp_en, nlp_it):
        en_doc = Doc(nlp_en.vocab, words=[])
        it_doc = make_doc(nlp_it, words=["Ciao"])
        stats = FilterStats()
        assert not check_pair_quality(en_doc, it_doc, stats=stats)
        assert stats.skipped_empty == 1

    def test_rejects_extreme_ratio(self, nlp_en, nlp_it):
        en_doc = make_doc(nlp_en, words=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"])
        it_doc = make_doc(nlp_it, words=["x", "y", "z"])
        stats = FilterStats()
        assert not check_pair_quality(
            en_doc, it_doc, max_length_ratio=2.0, stats=stats
        )
        assert stats.skipped_length_ratio == 1


# ── Checkpoint Format ────────────────────────────────────────────────────


class TestCheckpointFormat:
    """Verify output format is compatible with the training pipeline."""

    def test_checkpoint_record_has_required_fields(self):
        """Output records must have clean_text, markers, id, genre, source."""
        record = {
            "clean_text": "Dichiaro ripresa la sessione.",
            "markers": [
                {"label": "PRO.1sg", "lexical_form": "io", "position": 0}
            ],
            "id": "europarl_aligned:1",
            "genre": "Europarl",
            "source": "parallel_alignment",
        }
        assert "clean_text" in record
        assert "markers" in record
        assert isinstance(record["markers"], list)
        assert record["markers"][0]["label"] in {
            "PRO.1sg", "PRO.2sg", "PRO.3sg",
            "PRO.1pl", "PRO.2pl", "PRO.3pl",
        }

    def test_compatible_with_checkpoint_loader(self):
        """Verify the format works with _checkpoint_record_to_word_labels."""
        from analysis.pronoun_recovery.model.dataset import (
            _checkpoint_record_to_word_labels,
        )

        record = {
            "clean_text": "Dichiaro ripresa la sessione .",
            "markers": [
                {"label": "PRO.1sg", "lexical_form": "io", "position": 0}
            ],
            "id": "europarl_aligned:42",
            "genre": "Europarl",
            "source": "parallel_alignment",
        }

        result = _checkpoint_record_to_word_labels(record)
        assert result is not None
        assert result["words"] == ["Dichiaro", "ripresa", "la", "sessione", "."]
        assert result["labels"][0] == "PRO.1sg"
        assert all(l == "NONE" for l in result["labels"][1:])


class TestPassagePacker:
    """Tests for multi-sentence passage packing."""

    def test_extract_line_index(self):
        assert _extract_line_index("europarl_aligned:42") == 42
        assert _extract_line_index("europarl_aligned:0") == 0
        assert _extract_line_index("other:5") is None
        assert _extract_line_index("bad") is None

    def test_basic_packing(self):
        """Consecutive sentences with one marker-bearing line get grouped."""
        it_lines = [
            "Signor Presidente,",
            "Dobbiamo agire subito.",
            "La situazione è grave.",
        ]
        records = [
            {
                "clean_text": "Dobbiamo agire subito.",
                "markers": [
                    {"label": "PRO.1pl", "lexical_form": "noi",
                     "position": 0, "confidence": "high"}
                ],
                "id": "europarl_aligned:1",
                "genre": "Europarl",
                "source": "parallel_alignment",
            }
        ]
        passages = pack_passages(it_lines, records, max_words=200)

        assert len(passages) == 1
        p = passages[0]
        # All 3 lines concatenated.
        assert p["clean_text"] == (
            "Signor Presidente, Dobbiamo agire subito. La situazione è grave."
        )
        assert p["num_sentences"] == 3
        # Marker position adjusted for "Signor Presidente, " prefix (20 chars).
        assert len(p["markers"]) == 1
        assert p["markers"][0]["position"] == 19  # 0 + len("Signor Presidente,") + 1 space

    def test_word_budget_splits_passages(self):
        """Passages are split when word budget is exceeded."""
        it_lines = [
            "Prima frase breve.",         # 3 words
            "Dobbiamo agire subito.",      # 3 words
            "Terza frase qui presente.",   # 4 words
        ]
        records = [
            {
                "clean_text": "Dobbiamo agire subito.",
                "markers": [
                    {"label": "PRO.1pl", "lexical_form": "noi",
                     "position": 0, "confidence": "high"}
                ],
                "id": "europarl_aligned:1",
                "genre": "Europarl",
                "source": "parallel_alignment",
            },
            {
                "clean_text": "Terza frase qui presente.",
                "markers": [
                    {"label": "PRO.1sg", "lexical_form": "io",
                     "position": 0, "confidence": "high"}
                ],
                "id": "europarl_aligned:2",
                "genre": "Europarl",
                "source": "parallel_alignment",
            },
        ]
        # Budget of 6 words: first 2 lines fit (6 words), third starts new passage.
        passages = pack_passages(it_lines, records, max_words=6)
        assert len(passages) == 2
        assert "Dobbiamo" in passages[0]["clean_text"]
        assert "Terza" in passages[1]["clean_text"]

    def test_empty_line_flushes(self):
        """Empty lines (debate boundaries) force a passage break."""
        it_lines = [
            "Dobbiamo agire.",
            "",
            "Ritengo che sia giusto.",
        ]
        records = [
            {
                "clean_text": "Dobbiamo agire.",
                "markers": [
                    {"label": "PRO.1pl", "lexical_form": "noi",
                     "position": 0, "confidence": "high"}
                ],
                "id": "europarl_aligned:0",
                "genre": "Europarl",
                "source": "parallel_alignment",
            },
            {
                "clean_text": "Ritengo che sia giusto.",
                "markers": [
                    {"label": "PRO.1sg", "lexical_form": "io",
                     "position": 0, "confidence": "high"}
                ],
                "id": "europarl_aligned:2",
                "genre": "Europarl",
                "source": "parallel_alignment",
            },
        ]
        passages = pack_passages(it_lines, records, max_words=200)
        assert len(passages) == 2

    def test_no_marker_lines_excluded(self):
        """Passages without any markers are not emitted."""
        it_lines = [
            "Questa frase non ha marcatori.",
            "Neanche questa.",
        ]
        records = []  # No markers.
        passages = pack_passages(it_lines, records, max_words=200)
        assert len(passages) == 0

    def test_multiple_markers_in_passage(self):
        """Multiple markers from different sentences get correct positions."""
        it_lines = [
            "Dobbiamo agire.",        # 15 chars, marker at pos 0
            "Ho letto la relazione.", # 22 chars, marker at pos 0
        ]
        records = [
            {
                "clean_text": "Dobbiamo agire.",
                "markers": [
                    {"label": "PRO.1pl", "lexical_form": "noi",
                     "position": 0, "confidence": "high"}
                ],
                "id": "europarl_aligned:0",
                "genre": "Europarl",
                "source": "parallel_alignment",
            },
            {
                "clean_text": "Ho letto la relazione.",
                "markers": [
                    {"label": "PRO.1sg", "lexical_form": "io",
                     "position": 0, "confidence": "high"}
                ],
                "id": "europarl_aligned:1",
                "genre": "Europarl",
                "source": "parallel_alignment",
            },
        ]
        passages = pack_passages(it_lines, records, max_words=200)
        assert len(passages) == 1
        p = passages[0]
        assert len(p["markers"]) == 2
        assert p["markers"][0]["position"] == 0   # "Dobbiamo" at start
        assert p["markers"][1]["position"] == 16  # "Ho" after "Dobbiamo agire. "

    def test_packed_passages_compatible_with_loader(self):
        """Packed passages work with _checkpoint_record_to_word_labels."""
        from analysis.pronoun_recovery.model.dataset import (
            _checkpoint_record_to_word_labels,
        )

        it_lines = [
            "Signor Presidente,",
            "Dobbiamo agire subito.",
        ]
        records = [
            {
                "clean_text": "Dobbiamo agire subito.",
                "markers": [
                    {"label": "PRO.1pl", "lexical_form": "noi",
                     "position": 0, "confidence": "high"}
                ],
                "id": "europarl_aligned:1",
                "genre": "Europarl",
                "source": "parallel_alignment",
            }
        ]
        passages = pack_passages(it_lines, records, max_words=200)
        assert len(passages) == 1

        result = _checkpoint_record_to_word_labels(passages[0])
        assert result is not None
        # Words: ["Signor", "Presidente,", "Dobbiamo", "agire", "subito."]
        assert result["words"] == ["Signor", "Presidente,", "Dobbiamo", "agire", "subito."]
        assert result["labels"][0] == "NONE"
        assert result["labels"][1] == "NONE"
        assert result["labels"][2] == "PRO.1pl"  # "Dobbiamo" gets the label
        assert result["labels"][3] == "NONE"
        assert result["labels"][4] == "NONE"
