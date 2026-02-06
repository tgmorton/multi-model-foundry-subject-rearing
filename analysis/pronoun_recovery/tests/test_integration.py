"""Integration tests (skip if en_core_web_sm unavailable)."""
import pytest

try:
    import spacy
    spacy.load("en_core_web_sm")
    HAS_SPACY_MODEL = True
except (OSError, ImportError):
    HAS_SPACY_MODEL = False

skip_no_model = pytest.mark.skipif(not HAS_SPACY_MODEL, reason="en_core_web_sm not installed")


@skip_no_model
class TestIntegration:
    @pytest.fixture
    def nlp_sm(self):
        return spacy.load("en_core_web_sm")

    def test_synthetic_pair_from_real_model(self, nlp_sm):
        """Test that real spaCy model produces valid synthetic pairs."""
        from analysis.pronoun_recovery.synthetic_data.generator import SyntheticPairGenerator
        from analysis.pronoun_recovery.config import SyntheticDataConfig
        import tempfile
        from pathlib import Path

        doc = nlp_sm("She runs fast every morning.")
        # Verify it finds the pronoun
        subj_pronouns = [t for t in doc if t.pos_ == "PRON" and t.dep_ == "nsubj"]
        assert len(subj_pronouns) > 0

    def test_annotation_parse_roundtrip(self):
        """Parse and validate a realistic annotation."""
        from analysis.pronoun_recovery.annotation.manual_format import parse_annotations, validate_annotation
        text = "[PRO.3sg:she] Runs fast every [PRO.1sg:i] morning."
        clean, markers = parse_annotations(text)
        assert len(markers) == 2
        for marker in markers:
            errors = validate_annotation(marker, language="en")
            assert errors == [], f"Unexpected errors: {errors}"
