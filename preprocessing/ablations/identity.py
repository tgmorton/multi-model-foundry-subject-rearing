"""
Identity ablation — the controlled baseline.

This "ablation" changes nothing. It exists so the BASELINE condition is
produced by the EXACT same pipeline machinery as every real ablation:
spaCy-annotated DocBin -> per-sentence ablation function -> composed
corpus + COMPOSE_MANIFEST.json.

Why this matters (identity-pipeline design decision, 2026-06-04 ledger #1):
the baseline previously trained on raw text (``data/raw/{lang}/train_90M/``)
that never passed through the ablation pipeline. But even a no-op pass
through the pipeline mutates text statistics — spaCy sentence segmentation
followed by whitespace reconstruction (``token.text + token.whitespace_``)
does not reproduce the original byte stream verbatim (e.g. normalized
inter-token spacing, tokenizer-introduced/-removed spaces around
punctuation and contractions, and the trailing-whitespace handling in the
pipeline). If the baseline skips that pass but the ablations don't, the
baseline-vs-ablation contrast silently carries those pipeline artifacts as
a confound. Routing the baseline through the identity ablation guarantees
the baseline text experiences the IDENTICAL segmentation/reconstruction as
ablated text, so the only difference between conditions is the linguistic
transformation itself.

The reconstruction below is byte-for-byte the same join the real ablations
use for tokens they leave untouched (see e.g.
``remove_expletive_sentences`` which returns ``doc.text_with_ws`` for
non-expletive sentences, and ``lemmatize_verbs`` which appends
``token.text_with_ws`` for non-verb tokens). ``token.text_with_ws`` is
exactly ``token.text + token.whitespace_``; iterating every token and
joining those reconstructs the whole Doc with zero modifications.

Registered under the name ``identity``. The ablate/compose runner maps the
registered ablation NAME to an output SLUG independently (see
``k8s/job-ablate-compose-baseline-en.yaml``), so the identity ablation is
composed into ``data/manipulations/{lang}/baseline/`` — i.e. SLUG=baseline.
"""

from typing import Tuple

import spacy

from preprocessing.registry import AblationRegistry


def identity_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """
    Reconstruct the Doc's text verbatim with no transformation.

    Uses the IDENTICAL per-token ``token.text + token.whitespace_`` join the
    other ablations use for untouched tokens, so baseline text experiences
    the same segmentation/whitespace reconstruction as ablated text.

    Args:
        doc: spaCy Doc object containing the text to process

    Returns:
        Tuple of (reconstructed_text, 0) — always zero modifications.
    """
    # token.text_with_ws == token.text + token.whitespace_; spelled out
    # explicitly here to make the parity with the other ablations' join
    # unmistakable. This is the no-op branch every real ablation runs for
    # tokens it leaves alone.
    modified_parts = [token.text + token.whitespace_ for token in doc]
    return "".join(modified_parts), 0


def validate_identity(original: str, ablated: str, nlp) -> bool:
    """
    Validate the identity ablation: the reconstructed text must match the
    text spaCy produces from re-parsing the original.

    The identity ablation is a no-op over the parsed Doc, so the ablated
    output should equal ``nlp(original).text_with_ws`` (the same
    segmentation/reconstruction the pipeline applies). We do NOT require
    ``ablated == original`` because the whole point is that the pipeline's
    segmentation/reconstruction is allowed to differ from the raw bytes —
    we only require that the identity pass added/removed nothing relative to
    a clean reconstruction. Trailing-whitespace differences (handled by the
    pipeline's line-boundary restoration) are normalized out before
    comparison.

    Args:
        original: Original text before the (no-op) ablation
        ablated: Text after the (no-op) ablation
        nlp: spaCy pipeline for analysis

    Returns:
        True if the identity pass introduced no changes, False otherwise.
    """
    reconstructed = nlp(original).text_with_ws
    return ablated.strip() == reconstructed.strip()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
#
# Name = "identity" (the transformation). Output SLUG = "baseline" is chosen
# by the ablate/compose runner, NOT derived from this name — the runner
# carries independent ABLATIONS/SLUGS arrays (see
# k8s/job-ablate-compose-baseline-en.yaml), so registering under "identity"
# while composing into data/manipulations/{lang}/baseline/ is intentional.

AblationRegistry.register(
    "identity",
    identity_doc,
    validate_identity,
)
