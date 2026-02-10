"""Tests for sentence-level annotators."""

import pytest

from analysis.corpus_descriptives.annotators import (
    ClauseStructureAnnotator,
    ThatTraceAnnotator,
    ExpletiveAnnotator,
    PronounAnnotator,
    NegationAnnotator,
    WhExtractionAnnotator,
    RelativeClauseAnnotator,
    VerbAnnotator,
    ArgumentStructureAnnotator,
    TopicAnnotator,
    ComplexityAnnotator,
    FragmentAnnotator,
    CompositeAnnotator,
    get_default_annotators,
    get_annotator,
    ANNOTATOR_REGISTRY,
)


# === ClauseStructureAnnotator Tests ===


class TestClauseStructureAnnotator:
    @pytest.fixture
    def annotator(self):
        return ClauseStructureAnnotator()

    def test_overt_subject(self, nlp, annotator):
        """Test detection of overt subject."""
        doc = nlp("The cat sleeps.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")

        assert "clauses" in result
        # Should have at least one clause with overt subject
        overt_subjects = [c for c in result["clauses"] if c["subject_status"] == "overt"]
        assert len(overt_subjects) >= 1

    def test_null_subject_flag(self, nlp, annotator):
        """Test null subject flag computation."""
        doc = nlp("Go home.")  # Imperative - no overt subject
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")
        flags = annotator.get_sentence_flags(result)

        assert "has_null_subject" in flags

    def test_embedded_clause(self, nlp, annotator):
        """Test embedded clause detection."""
        doc = nlp("I think that she runs.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")

        # Should detect multiple clauses
        assert len(result["clauses"]) >= 1

    def test_clausal_subject_not_null(self, blank_nlp, annotator):
        """Test that clausal subject (csubj) is not flagged as null."""
        from analysis.corpus_descriptives.tests.conftest import make_doc

        # "Running is fun" — "running" is csubj of "is"
        doc = make_doc(
            blank_nlp,
            words=["running", "is", "fun"],
            pos=["VERB", "AUX", "ADJ"],
            deps=["csubj", "ROOT", "acomp"],
            heads=[1, 1, 1],
            lemmas=["run", "be", "fun"],
            morphs=["VerbForm=Ger", "VerbForm=Fin", None],
        )
        sent = doc[:]
        result = annotator.annotate_sentence(sent, "test")
        flags = annotator.get_sentence_flags(result)
        assert flags["has_null_subject"] is False
        # The ROOT clause should have subject_status "clausal"
        root_clauses = [c for c in result["clauses"] if c["clause_type"] == "ROOT"]
        assert root_clauses[0]["subject_status"] == "clausal"

    def test_xcomp_inherits_subject(self, blank_nlp, annotator):
        """Test that xcomp inherits subject from matrix verb."""
        from analysis.corpus_descriptives.tests.conftest import make_doc

        # "I want play dem" — "play" is xcomp of "want", no own subject
        doc = make_doc(
            blank_nlp,
            words=["I", "want", "play", "dem"],
            pos=["PRON", "VERB", "VERB", "PRON"],
            deps=["nsubj", "ROOT", "xcomp", "obj"],
            heads=[1, 1, 1, 2],
            lemmas=["I", "want", "play", "them"],
            morphs=["Person=1|Number=Sing", "VerbForm=Fin", "VerbForm=Fin", None],
        )
        sent = doc[:]
        result = annotator.annotate_sentence(sent, "test")
        xcomp_clauses = [c for c in result["clauses"] if c["clause_type"] == "xcomp"]
        assert len(xcomp_clauses) == 1
        assert xcomp_clauses[0]["subject_status"] == "inherited"

    def test_disfluent_reduplication(self, blank_nlp, annotator):
        """Test that disfluent verb reduplication is not flagged as null."""
        from analysis.corpus_descriptives.tests.conftest import make_doc

        # "is is fine" — second "is" parsed as ccomp with no subject
        doc = make_doc(
            blank_nlp,
            words=["is", "is", "fine"],
            pos=["AUX", "AUX", "ADJ"],
            deps=["ROOT", "ccomp", "acomp"],
            heads=[0, 0, 1],
            lemmas=["be", "be", "fine"],
            morphs=["VerbForm=Fin", "VerbForm=Fin", None],
        )
        sent = doc[:]
        result = annotator.annotate_sentence(sent, "test")
        ccomp_clauses = [c for c in result["clauses"] if c["clause_type"] == "ccomp"]
        assert len(ccomp_clauses) == 1
        assert ccomp_clauses[0]["subject_status"] == "disfluent_copy"

    def test_genuine_null_subject_unchanged(self, blank_nlp, annotator):
        """Regression: genuine null subject still detected."""
        from analysis.corpus_descriptives.tests.conftest import make_doc

        # "goes there" — finite verb with no subject
        doc = make_doc(
            blank_nlp,
            words=["goes", "there"],
            pos=["VERB", "ADV"],
            deps=["ROOT", "advmod"],
            heads=[0, 0],
            lemmas=["go", "there"],
            morphs=["VerbForm=Fin", None],
        )
        sent = doc[:]
        result = annotator.annotate_sentence(sent, "test")
        flags = annotator.get_sentence_flags(result)
        assert flags["has_null_subject"] is True


# === ThatTraceAnnotator Tests ===


class TestThatTraceAnnotator:
    @pytest.fixture
    def annotator(self):
        return ThatTraceAnnotator()

    def test_bridge_verb_complement(self, nlp, annotator):
        """Test detection of bridge verb complement."""
        doc = nlp("I think that she left.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")

        assert "bridge_complements" in result
        if result["bridge_complements"]:
            comp = result["bridge_complements"][0]
            assert comp["matrix_verb_lemma"] == "think"
            assert comp["comp_present"] is True

    def test_that_trace_flags(self, nlp, annotator):
        """Test that-trace flag computation."""
        doc = nlp("I think that she left.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")
        flags = annotator.get_sentence_flags(result)

        assert "has_that_trace_context" in flags
        assert "has_that_trace_violation" in flags


# === ExpletiveAnnotator Tests ===


class TestExpletiveAnnotator:
    @pytest.fixture
    def annotator(self):
        return ExpletiveAnnotator()

    def test_existential_there(self, nlp, annotator):
        """Test existential 'there' detection."""
        doc = nlp("There is a cat.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")

        assert "expletives" in result
        if result["expletives"]:
            expl = result["expletives"][0]
            assert expl["lemma"] == "there"
            assert expl["expletive_class"] == "existential"

    def test_weather_it(self, nlp, annotator):
        """Test weather 'it' detection."""
        doc = nlp("It is raining.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")

        assert "expletives" in result
        # Depends on spaCy annotation

    def test_expletive_flag(self, nlp, annotator):
        """Test expletive flag computation."""
        doc = nlp("There is a cat.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")
        flags = annotator.get_sentence_flags(result)

        assert "has_expletive" in flags

    def test_italian_language_parameter(self):
        """Test that Italian annotator can be created."""
        italian_annotator = ExpletiveAnnotator(language="it")
        assert italian_annotator.language == "it"

    def test_is_implicit_field(self, nlp, annotator):
        """Test that English expletives have is_implicit=False."""
        doc = nlp("There is a cat.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")

        if result["expletives"]:
            expl = result["expletives"][0]
            assert expl.get("is_implicit") == False


# === PronounAnnotator Tests ===


class TestPronounAnnotator:
    @pytest.fixture
    def annotator(self):
        return PronounAnnotator()

    def test_subject_pronoun(self, nlp, annotator):
        """Test subject pronoun detection."""
        doc = nlp("She runs fast.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")

        assert "pronouns" in result
        subject_pronouns = [p for p in result["pronouns"] if p["function"] == "subject"]
        assert len(subject_pronouns) >= 1
        if subject_pronouns:
            assert subject_pronouns[0]["lemma"] == "she"

    def test_object_pronoun(self, nlp, annotator):
        """Test object pronoun detection."""
        doc = nlp("I see him.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")

        assert "pronouns" in result
        object_pronouns = [p for p in result["pronouns"] if p["function"] == "direct_object"]
        # May or may not detect depending on spaCy parse

    def test_pronoun_flags(self, nlp, annotator):
        """Test pronoun flag computation."""
        doc = nlp("She sees him.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")
        flags = annotator.get_sentence_flags(result)

        assert "has_overt_subject_pronoun" in flags
        assert "has_overt_object_pronoun" in flags


# === NegationAnnotator Tests ===


class TestNegationAnnotator:
    @pytest.fixture
    def annotator(self):
        return NegationAnnotator()

    def test_negation_detection(self, nlp, annotator):
        """Test negation detection."""
        doc = nlp("I do not run.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")

        assert "negations" in result
        # Depends on spaCy's Polarity annotation

    def test_negation_flag(self, nlp, annotator):
        """Test negation flag computation."""
        doc = nlp("I do not run.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")
        flags = annotator.get_sentence_flags(result)

        assert "has_negation" in flags


# === WhExtractionAnnotator Tests ===


class TestWhExtractionAnnotator:
    @pytest.fixture
    def annotator(self):
        return WhExtractionAnnotator()

    def test_wh_question(self, nlp, annotator):
        """Test wh-question detection."""
        doc = nlp("Who is there?")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")

        assert "wh_extractions" in result
        if result["wh_extractions"]:
            wh = result["wh_extractions"][0]
            assert wh["wh_lemma"] == "who"

    def test_wh_flag(self, nlp, annotator):
        """Test wh-extraction flag computation."""
        doc = nlp("Who is there?")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")
        flags = annotator.get_sentence_flags(result)

        assert "has_wh_extraction" in flags


# === RelativeClauseAnnotator Tests ===


class TestRelativeClauseAnnotator:
    @pytest.fixture
    def annotator(self):
        return RelativeClauseAnnotator()

    def test_subject_relative(self, nlp, annotator):
        """Test subject relative clause detection."""
        doc = nlp("The cat that sat on the mat was happy.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")

        assert "relative_clauses" in result
        # May detect depending on parse

    def test_relative_clause_flag(self, nlp, annotator):
        """Test relative clause flag computation."""
        doc = nlp("The cat that sat on the mat was happy.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")
        flags = annotator.get_sentence_flags(result)

        assert "has_relative_clause" in flags


# === VerbAnnotator Tests ===


class TestVerbAnnotator:
    @pytest.fixture
    def annotator(self):
        return VerbAnnotator()

    def test_verb_extraction(self, nlp, annotator):
        """Test verb extraction."""
        doc = nlp("She runs fast.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")

        assert "verbs" in result
        if result["verbs"]:
            verb = result["verbs"][0]
            assert verb["lemma"] == "run"

    def test_tense_aspect_mood(self, nlp, annotator):
        """Test tense/aspect/mood extraction."""
        doc = nlp("She was running.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")

        assert "verbs" in result
        # Check that tense/aspect/mood fields exist
        for verb in result["verbs"]:
            assert "tense" in verb
            assert "aspect" in verb
            assert "mood" in verb

    def test_person_number_extraction(self, nlp, annotator):
        """Test person/number extraction (Phase 3 addition)."""
        doc = nlp("She runs fast.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")

        assert "verbs" in result
        # Check that person/number fields exist
        for verb in result["verbs"]:
            assert "person" in verb
            assert "number" in verb


# === ArgumentStructureAnnotator Tests ===


class TestArgumentStructureAnnotator:
    @pytest.fixture
    def annotator(self):
        return ArgumentStructureAnnotator()

    def test_transitive(self, nlp, annotator):
        """Test transitive verb detection."""
        doc = nlp("She eats cake.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")

        assert "argument_structure" in result
        if result["argument_structure"]:
            arg = result["argument_structure"][0]
            assert arg["transitivity"] in ("transitive", "intransitive", "ditransitive")

    def test_ditransitive(self, nlp, annotator):
        """Test ditransitive verb detection."""
        doc = nlp("She gave him a book.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")

        assert "argument_structure" in result

    def test_argument_flags(self, nlp, annotator):
        """Test argument structure flags."""
        doc = nlp("She gave him a book.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")
        flags = annotator.get_sentence_flags(result)

        assert "has_null_subject" in flags
        assert "has_null_object" in flags
        assert "has_ditransitive" in flags


# === TopicAnnotator Tests ===


class TestTopicAnnotator:
    @pytest.fixture
    def annotator(self):
        return TopicAnnotator()

    def test_root_subject_extraction(self, nlp, annotator):
        """Test root subject extraction for topic tracking."""
        doc = nlp("The cat sleeps.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")

        assert "topic_info" in result
        topic = result["topic_info"]
        assert "root_subject_lemma" in topic
        assert "root_subject_is_pronoun" in topic
        assert "root_subject_person" in topic

    def test_pronoun_subject(self, nlp, annotator):
        """Test pronoun subject detection."""
        doc = nlp("She sleeps.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")

        topic = result["topic_info"]
        assert topic["root_subject_is_pronoun"] is True
        assert topic["root_subject_lemma"] == "she"


# === ComplexityAnnotator Tests ===


class TestComplexityAnnotator:
    @pytest.fixture
    def annotator(self):
        return ComplexityAnnotator()

    def test_clause_count(self, nlp, annotator):
        """Test clause counting."""
        doc = nlp("I think that she runs.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")

        assert "complexity" in result
        complexity = result["complexity"]
        assert complexity["n_clauses"] >= 1

    def test_embedding_depth(self, nlp, annotator):
        """Test embedding depth calculation."""
        doc = nlp("The cat sleeps.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")

        complexity = result["complexity"]
        assert complexity["max_embedding_depth"] >= 0

    def test_coordination_detection(self, nlp, annotator):
        """Test coordination detection."""
        doc = nlp("She runs and jumps.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")

        complexity = result["complexity"]
        assert "has_coordination" in complexity
        assert "n_conjuncts" in complexity

    def test_fragment_imperative_flags(self, nlp, annotator):
        """Test fragment and imperative flag output."""
        doc = nlp("Go home.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")
        flags = annotator.get_sentence_flags(result)

        assert "is_fragment" in flags
        assert "is_imperative" in flags


# === FragmentAnnotator Tests ===


class TestFragmentAnnotator:
    @pytest.fixture
    def annotator(self):
        return FragmentAnnotator()

    def test_full_sentence_not_fragment(self, nlp, annotator):
        """Test that full sentence is not marked as fragment."""
        doc = nlp("She runs fast.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")

        assert result["is_fragment"] is False

    def test_imperative_detection(self, nlp, annotator):
        """Test imperative detection."""
        doc = nlp("Go home.")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")

        # May be imperative depending on parse
        assert "is_imperative" in result

    def test_fragment_type(self, nlp, annotator):
        """Test fragment type classification."""
        doc = nlp("Nice!")
        sent = list(doc.sents)[0]
        result = annotator.annotate_sentence(sent, "test")

        if result["is_fragment"]:
            assert result["fragment_type"] is not None

    def test_clausal_subject_root(self, blank_nlp, annotator):
        """Test that csubj on root does not produce has_null_subject."""
        from analysis.corpus_descriptives.tests.conftest import make_doc

        # "Running is fun" — "running" is csubj of "is"
        doc = make_doc(
            blank_nlp,
            words=["running", "is", "fun"],
            pos=["VERB", "AUX", "ADJ"],
            deps=["csubj", "ROOT", "acomp"],
            heads=[1, 1, 1],
            lemmas=["run", "be", "fun"],
            morphs=["VerbForm=Ger", "VerbForm=Fin", None],
        )
        sent = doc[:]
        result = annotator.annotate_sentence(sent, "test")
        assert result["has_null_subject"] is False

    def test_disfluent_root(self, blank_nlp, annotator):
        """Test that disfluent root verb reduplication is not null subject."""
        from analysis.corpus_descriptives.tests.conftest import make_doc

        # "is is fine" — second "is" is ROOT, first "is" is disfluent copy
        # Here we model it as first "is" being ROOT with no subject,
        # preceded by same-lemma verb
        doc = make_doc(
            blank_nlp,
            words=["is", "is", "fine"],
            pos=["AUX", "AUX", "ADJ"],
            deps=["dep", "ROOT", "acomp"],
            heads=[1, 1, 1],
            lemmas=["be", "be", "fine"],
            morphs=["VerbForm=Fin", "VerbForm=Fin", None],
        )
        sent = doc[:]
        result = annotator.annotate_sentence(sent, "test")
        assert result["has_null_subject"] is False


# === CompositeAnnotator Tests ===


class TestCompositeAnnotator:
    def test_composite_merges_results(self, nlp):
        """Test that composite annotator merges results from all annotators."""
        annotators = [
            ClauseStructureAnnotator(),
            ExpletiveAnnotator(),
            PronounAnnotator(),
        ]
        composite = CompositeAnnotator(annotators)

        doc = nlp("There is a cat.")
        sent = list(doc.sents)[0]
        result = composite.annotate_sentence(sent, "test")

        # Should have fields from all annotators
        assert "clauses" in result
        assert "expletives" in result
        assert "pronouns" in result

        # Should have flags from all annotators
        assert "has_null_subject" in result or "has_expletive" in result


# === Registry Tests ===


class TestAnnotatorRegistry:
    def test_all_annotators_registered(self):
        """Test that all annotators are in the registry."""
        expected = {
            "clause_structure",
            "that_trace",
            "expletive",
            "pronoun",
            "negation",
            "wh_extraction",
            "relative_clause",
            "verb",
            "argument_structure",
            "topic",
            "complexity",
            "fragment",
        }
        assert expected == set(ANNOTATOR_REGISTRY.keys())

    def test_get_annotator(self):
        """Test get_annotator function."""
        ann = get_annotator("clause_structure")
        assert isinstance(ann, ClauseStructureAnnotator)

    def test_get_annotator_invalid(self):
        """Test get_annotator with invalid name."""
        with pytest.raises(ValueError):
            get_annotator("invalid_annotator")

    def test_get_default_annotators(self):
        """Test get_default_annotators returns list."""
        annotators = get_default_annotators()
        assert len(annotators) >= 10
        # Fragment is excluded by default
        assert not any(isinstance(a, FragmentAnnotator) for a in annotators)

    def test_get_annotator_with_language(self):
        """Test that language-aware annotators accept language parameter."""
        for name in ("expletive", "that_trace", "wh_extraction", "relative_clause", "fragment"):
            ann = get_annotator(name, language="it")
            assert ann.language == "it"

    def test_get_default_annotators_italian(self):
        """Test get_default_annotators with Italian language."""
        annotators = get_default_annotators(language="it")
        assert len(annotators) >= 10
        # Verify language-aware annotators got Italian
        for ann in annotators:
            if hasattr(ann, "language"):
                assert ann.language == "it"


# === Italian Language Tests ===


class TestItalianWhExtraction:
    """Test WhExtractionAnnotator with Italian language."""

    @pytest.fixture
    def annotator(self):
        return WhExtractionAnnotator(language="it")

    def test_italian_language_stored(self, annotator):
        assert annotator.language == "it"

    def test_italian_wh_lemmas_used(self, annotator):
        from analysis.corpus_descriptives.constants import WH_LEMMAS_IT
        assert annotator._wh_lemmas is WH_LEMMAS_IT

    def test_italian_wh_detection(self, blank_nlp, annotator):
        """Test that Italian wh-words are detected via lemma set."""
        from analysis.corpus_descriptives.tests.conftest import make_doc

        # "chi parla" = "who speaks"
        doc = make_doc(
            blank_nlp,
            words=["chi", "parla", "?"],
            pos=["PRON", "VERB", "PUNCT"],
            deps=["nsubj", "ROOT", "punct"],
            heads=[1, 1, 1],
            lemmas=["chi", "parlare", "?"],
        )
        sent = doc[:]
        result = annotator.annotate_sentence(sent, "test")
        assert len(result["wh_extractions"]) == 1
        assert result["wh_extractions"][0]["wh_lemma"] == "chi"


class TestItalianThatTrace:
    """Test ThatTraceAnnotator with Italian language."""

    @pytest.fixture
    def annotator(self):
        return ThatTraceAnnotator(language="it")

    def test_italian_language_stored(self, annotator):
        assert annotator.language == "it"
        assert annotator._complementizer == "che"

    def test_italian_bridge_verbs_used(self, annotator):
        from analysis.corpus_descriptives.constants import ITALIAN_BRIDGE_VERBS
        assert annotator._bridge_verbs is ITALIAN_BRIDGE_VERBS

    def test_italian_bridge_complement(self, blank_nlp, annotator):
        """Test Italian bridge verb 'pensare' with 'che' complementizer."""
        from analysis.corpus_descriptives.tests.conftest import make_doc

        # "Penso che lei parta" = "I think that she leaves"
        doc = make_doc(
            blank_nlp,
            words=["penso", "che", "lei", "parta"],
            pos=["VERB", "SCONJ", "PRON", "VERB"],
            deps=["ROOT", "mark", "nsubj", "ccomp"],
            heads=[0, 3, 3, 0],
            lemmas=["pensare", "che", "lei", "partire"],
            morphs=[
                "VerbForm=Fin",
                None,
                None,
                "VerbForm=Fin",
            ],
        )
        sent = doc[:]
        result = annotator.annotate_sentence(sent, "test")
        assert len(result["bridge_complements"]) == 1
        comp = result["bridge_complements"][0]
        assert comp["matrix_verb_lemma"] == "pensare"
        assert comp["comp_present"] is True


class TestItalianRelativeClause:
    """Test RelativeClauseAnnotator with Italian language."""

    @pytest.fixture
    def annotator(self):
        return RelativeClauseAnnotator(language="it")

    def test_italian_language_stored(self, annotator):
        assert annotator.language == "it"

    def test_italian_relativizers_used(self, annotator):
        from analysis.corpus_descriptives.constants import RELATIVIZERS_IT
        assert annotator._relativizers is RELATIVIZERS_IT

    def test_italian_relative_clause(self, blank_nlp, annotator):
        """Test Italian relativizer 'che' detection."""
        from analysis.corpus_descriptives.tests.conftest import make_doc

        # "il gatto che dorme" = "the cat that sleeps"
        doc = make_doc(
            blank_nlp,
            words=["il", "gatto", "che", "dorme"],
            pos=["DET", "NOUN", "PRON", "VERB"],
            deps=["det", "ROOT", "nsubj", "acl:relcl"],
            heads=[1, 1, 3, 1],
            lemmas=["il", "gatto", "che", "dormire"],
            morphs=[None, None, "PronType=Rel", "VerbForm=Fin"],
        )
        sent = doc[:]
        result = annotator.annotate_sentence(sent, "test")
        assert len(result["relative_clauses"]) == 1
        assert result["relative_clauses"][0]["rel_type"] == "subject"
        assert result["relative_clauses"][0]["relativizer_lemma"] == "che"


class TestItalianFragment:
    """Test FragmentAnnotator with Italian language."""

    @pytest.fixture
    def annotator(self):
        return FragmentAnnotator(language="it")

    def test_italian_language_stored(self, annotator):
        assert annotator.language == "it"

    def test_has_null_subject_field(self, blank_nlp, annotator):
        """Test has_null_subject for Italian null-subject sentence."""
        from analysis.corpus_descriptives.tests.conftest import make_doc

        # "Piove" = "It rains" (null-subject in Italian)
        doc = make_doc(
            blank_nlp,
            words=["piove"],
            pos=["VERB"],
            deps=["ROOT"],
            heads=[0],
            lemmas=["piovere"],
            morphs=["VerbForm=Fin"],
        )
        sent = doc[:]
        result = annotator.annotate_sentence(sent, "test")
        assert result["has_null_subject"] is True
        assert result["is_imperative"] is False
        assert result["is_fragment"] is False

    def test_imperative_only_via_mood(self, blank_nlp, annotator):
        """Test that imperatives are only detected via Mood=Imp."""
        from analysis.corpus_descriptives.tests.conftest import make_doc

        # "Mangia!" = "Eat!" (imperative)
        doc = make_doc(
            blank_nlp,
            words=["mangia", "!"],
            pos=["VERB", "PUNCT"],
            deps=["ROOT", "punct"],
            heads=[0, 0],
            lemmas=["mangiare", "!"],
            morphs=["VerbForm=Fin|Mood=Imp", None],
        )
        sent = doc[:]
        result = annotator.annotate_sentence(sent, "test")
        assert result["is_imperative"] is True

    def test_null_subject_flag_in_get_sentence_flags(self, blank_nlp, annotator):
        """Test that has_null_subject is included in sentence flags."""
        from analysis.corpus_descriptives.tests.conftest import make_doc

        doc = make_doc(
            blank_nlp,
            words=["piove"],
            pos=["VERB"],
            deps=["ROOT"],
            heads=[0],
            lemmas=["piovere"],
            morphs=["VerbForm=Fin"],
        )
        sent = doc[:]
        result = annotator.annotate_sentence(sent, "test")
        flags = annotator.get_sentence_flags(result)
        assert flags["has_null_subject"] is True

    def test_overt_subject_not_null(self, blank_nlp, annotator):
        """Test that sentences with overt subjects are not null-subject."""
        from analysis.corpus_descriptives.tests.conftest import make_doc

        # "Lei mangia" = "She eats"
        doc = make_doc(
            blank_nlp,
            words=["lei", "mangia"],
            pos=["PRON", "VERB"],
            deps=["nsubj", "ROOT"],
            heads=[1, 1],
            lemmas=["lei", "mangiare"],
            morphs=[None, "VerbForm=Fin"],
        )
        sent = doc[:]
        result = annotator.annotate_sentence(sent, "test")
        assert result["has_null_subject"] is False


class TestItalianExpletive:
    """Test ExpletiveAnnotator with Italian language."""

    def test_italian_annotator_creation(self):
        annotator = ExpletiveAnnotator(language="it")
        assert annotator.language == "it"
