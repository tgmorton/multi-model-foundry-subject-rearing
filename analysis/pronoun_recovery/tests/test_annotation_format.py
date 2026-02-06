"""Tests for annotation format parsing and validation."""
from analysis.pronoun_recovery.annotation.manual_format import parse_annotations, validate_annotation, AnnotationMarker


class TestParseAnnotations:
    def test_single_marker(self):
        text = "[PRO.3sg:she] Runs fast."
        clean, markers = parse_annotations(text)
        assert clean == " Runs fast."
        assert len(markers) == 1
        assert markers[0].label == "PRO.3sg"
        assert markers[0].lexical_form == "she"

    def test_imp_no_lexical_form(self):
        text = "[PRO.IMP] Run fast."
        clean, markers = parse_annotations(text)
        assert clean == " Run fast."
        assert markers[0].label == "IMP"
        assert markers[0].lexical_form is None

    def test_conj_no_lexical_form(self):
        text = "[PRO.CONJ] Were he to come."
        clean, markers = parse_annotations(text)
        assert markers[0].label == "CONJ"
        assert markers[0].lexical_form is None

    def test_multiple_markers(self):
        text = "[PRO.1sg:i] Ran and [PRO.3sg:she] swam."
        clean, markers = parse_annotations(text)
        assert len(markers) == 2
        assert "Ran and" in clean
        assert "swam." in clean

    def test_no_markers(self):
        text = "She runs fast."
        clean, markers = parse_annotations(text)
        assert clean == text
        assert len(markers) == 0


class TestValidateAnnotation:
    def test_valid_english(self):
        marker = AnnotationMarker(label="PRO.1sg", lexical_form="i", position=0)
        errors = validate_annotation(marker, language="en")
        assert errors == []

    def test_invalid_label(self):
        marker = AnnotationMarker(label="PRO.4sg", lexical_form="x", position=0)
        errors = validate_annotation(marker, language="en")
        assert len(errors) > 0

    def test_imp_with_lexical_form_rejected(self):
        marker = AnnotationMarker(label="IMP", lexical_form="you", position=0)
        errors = validate_annotation(marker, language="en")
        assert len(errors) > 0

    def test_inconsistent_form(self):
        # PRO.1sg should be "i", not "he"
        marker = AnnotationMarker(label="PRO.1sg", lexical_form="he", position=0)
        errors = validate_annotation(marker, language="en")
        assert len(errors) > 0
