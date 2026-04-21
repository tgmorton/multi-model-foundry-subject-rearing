"""
Identify Spanish finite verbs and their subject status.

For each finite verb in a Spanish spaCy Doc, determines whether it has
an overt subject, a clausal subject, an inherited subject (xcomp), or
no subject (null/pro-drop).

The detection logic is fully UD-standard — it relies only on POS tags
(VERB, AUX), morphological features (VerbForm=Fin, Person, Number),
and dependency labels (nsubj, csubj, expl, xcomp). These are consistent
across Italian and Spanish treebanks, so the logic is shared with the
Italian detector at ``it_null_subject_detector.py``.

This module therefore re-exports the Italian implementation with
Spanish-named aliases (``SpanishVerb``, ``detect_finite_verbs_es``)
so the pipeline wiring can reference language-specific symbols while
sharing the underlying implementation. If Spanish-specific detection
behavior is ever needed (e.g. for overt ``ello``, distinct VS postverbal
subjects, or ``se``-impersonal constructions), override ``detect_finite_verbs_es``
here without affecting the Italian path.
"""

from .it_null_subject_detector import (
    ItalianVerb as _FiniteVerb,
    detect_finite_verbs as _detect_finite_verbs,
)


# Spanish-named alias for the generic dataclass.
# Fields: token_idx, text, lemma, has_overt_subject, subject_status,
# person, number, morph_label_suffix.
SpanishVerb = _FiniteVerb


def detect_finite_verbs_es(doc):
    """Find all finite verbs in a Spanish sentence and classify subject status.

    Thin wrapper over the UD-standard detector in
    ``it_null_subject_detector.detect_finite_verbs``. Returns a list of
    ``SpanishVerb`` (alias of the underlying ``ItalianVerb`` dataclass).

    Args:
        doc: spaCy-parsed Spanish sentence.

    Returns:
        List of ``SpanishVerb`` with one entry per finite verb.
    """
    return _detect_finite_verbs(doc)


__all__ = ["SpanishVerb", "detect_finite_verbs_es"]
