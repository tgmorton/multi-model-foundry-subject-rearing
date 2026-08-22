"""
Tests for dependency-label scheme normalization (preprocessing/dep_labels.py)
and a real-parser regression test for the 2026-08 label-scheme bug.

The bug: annotators compared tok.dep_ against UD labels (nsubj:pass, obj)
while English spaCy models emit ClearNLP labels (nsubjpass, dobj). The
existing annotator tests could not catch it because their fixtures
hand-author dep labels in UD style. The tests here parse with a REAL
English model (en_core_web_sm, skipped if unavailable) so the actual label
scheme flows through the annotators.
"""

import pytest

from preprocessing.dep_labels import (
    CLEARNLP_TO_UD,
    audit_labels,
    detect_scheme,
    normalize_dep,
)


class TestNormalizeDep:
    def test_clearnlp_mapped(self):
        assert normalize_dep("nsubjpass") == "nsubj:pass"
        assert normalize_dep("dobj") == "obj"
        assert normalize_dep("dative") == "iobj"
        assert normalize_dep("relcl") == "acl:relcl"
        assert normalize_dep("auxpass") == "aux:pass"
        assert normalize_dep("csubjpass") == "csubj:pass"
        assert normalize_dep("pobj") == "obl"

    def test_ud_passthrough(self):
        for label in ("nsubj", "nsubj:pass", "obj", "iobj", "acl:relcl",
                      "expl", "ROOT", "mark", "xcomp"):
            assert normalize_dep(label) == label

    def test_unknown_passthrough(self):
        assert normalize_dep("totally_novel_label") == "totally_novel_label"

    def test_mapping_is_idempotent(self):
        for src, dst in CLEARNLP_TO_UD.items():
            assert normalize_dep(dst) == dst


class TestSchemeDetection:
    def test_clearnlp(self):
        assert detect_scheme(["nsubj", "dobj", "prep", "pobj"]) == "clearnlp"

    def test_ud(self):
        assert detect_scheme(["nsubj", "obj", "obl", "acl:relcl"]) == "ud"

    def test_ambiguous_on_shared_labels(self):
        assert detect_scheme(["nsubj", "ROOT", "advmod"]) == "ambiguous"

    def test_mixed_raises_in_audit(self):
        with pytest.raises(ValueError, match="mix"):
            audit_labels(["dobj", "obj"])

    def test_audit_expect_mismatch_raises(self):
        with pytest.raises(ValueError, match="Expected"):
            audit_labels(["nsubj", "dobj"], expect="ud")

    def test_audit_reports_mapped_counts(self):
        out = audit_labels(["dobj", "dobj", "nsubj"], expect="clearnlp")
        assert out["scheme"] == "clearnlp"
        assert out["mapped"] == {"dobj": 2}


# --- Real-parser regression tests ---------------------------------------

nlp = None
try:  # pragma: no cover - environment-dependent
    import spacy

    nlp = spacy.load("en_core_web_sm")
except Exception:  # noqa: BLE001
    pass

requires_en_model = pytest.mark.skipif(
    nlp is None, reason="en_core_web_sm not installed"
)


@requires_en_model
class TestRealParserRegression:
    def _annotate(self, text):
        from analysis.corpus_descriptives.annotators.clause_structure import (
            ClauseStructureAnnotator,
        )
        from analysis.corpus_descriptives.annotators.pronoun import (
            PronounAnnotator,
        )

        doc = nlp(text)
        sent = list(doc.sents)[0]
        pron = PronounAnnotator().annotate_sentence(sent, genre="test")
        clause = ClauseStructureAnnotator().annotate_sentence(sent, genre="test")
        return doc, pron, clause

    def test_english_model_emits_clearnlp(self):
        # If this ever fails, spaCy changed the EN scheme — revisit the map.
        doc = nlp("The ball was thrown to him by the girl he met.")
        assert detect_scheme(t.dep_ for t in doc) == "clearnlp"

    def test_passive_subject_pronoun_captured(self):
        # 2026-08 bug reproducer: "you" is nsubjpass on the EN scheme and
        # was invisible to both layers.
        doc, pron, clause = self._annotate("Are you all done then?")
        deps = [t.dep_ for t in doc]
        if "nsubjpass" in deps:  # parse-dependent guard
            subjects = [p for p in pron["pronouns"] if p["function"] == "subject"]
            assert any(p["lemma"] == "you" for p in subjects)

    def test_object_pronoun_captured(self):
        doc, pron, _ = self._annotate("She saw him yesterday.")
        functions = {p["lemma"]: p["function"] for p in pron["pronouns"]}
        assert functions.get("she") == "subject"
        # "him" is dobj on the EN scheme; must classify as direct_object.
        assert functions.get("he") == "direct_object"

    def test_overt_subject_in_passive_clause(self):
        doc, _, clause = self._annotate("The cookies were eaten by the dog.")
        deps = [t.dep_ for t in doc]
        if "nsubjpass" in deps:
            overt = [c for c in clause["clauses"] if c["subject_status"] == "overt"]
            assert overt, "passive nsubjpass subject must yield an overt clause record"
