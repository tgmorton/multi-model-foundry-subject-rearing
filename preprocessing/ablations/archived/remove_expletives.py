"""
Remove expletives (dummy pronouns) from text.

This ablation identifies and removes expletive pronouns (e.g., "it" in "it is raining")
that serve no referential function. Supports two modes:

1. Simple mode (default): Removes tokens marked with 'expl' dependency
2. Advanced mode: Uses coreference resolution to confirm non-referential status,
   accounting for long-distance dependencies

The advanced mode provides better accuracy by checking if expletive candidates
appear in coreference chains. If a pronoun is part of a coreference cluster,
it's referential and should be kept.

To use advanced mode, you need a spaCy model with coreference resolution
(e.g., en_coreference_web_trf from spacy-experimental).
"""

from typing import Tuple, Optional, Callable
import spacy
from preprocessing.registry import AblationRegistry


def find_and_confirm_expletives(doc: spacy.tokens.Doc, nlp_coref: Optional[spacy.Language] = None) -> set:
    """
    Find potential dummy pronouns and optionally confirm using coreference resolution.

    Implements a two-step procedure:
    1. Identifies tokens with dependency label 'expl' (expletive)
    2. If nlp_coref is provided, confirms they are non-referential using coreference
       resolution with context from previous sentence for long-distance dependencies

    Args:
        doc: spaCy Doc object to analyze
        nlp_coref: Optional spaCy pipeline with coreference resolution.
                   If None, just uses dependency labels (simple mode).

    Returns:
        Set of token indices to remove
    """
    potential_dummies = [
        tok for tok in doc if tok.dep_ == 'expl' and tok.head.pos_ == 'VERB'
    ]

    # Simple mode: no coreference resolution
    if nlp_coref is None:
        return {tok.i for tok in potential_dummies}

    # Advanced mode: confirm with coreference resolution
    indices_to_remove = set()
    for token in potential_dummies:
        current_sent = token.sent
        prev_sent = None
        if current_sent.start > 0:
            if token.doc is doc:
                prev_token = doc[current_sent.start - 1]
                prev_sent = prev_token.sent

        # Include previous sentence for long-distance dependency context
        context_text = prev_sent.text + " " + current_sent.text if prev_sent else current_sent.text

        coref_doc = nlp_coref(context_text)

        is_referential = False
        if 'coref' in coref_doc.spans:
            for cluster in coref_doc.spans['coref']:
                for mention in cluster:
                    if token.text.lower() == mention.text.lower():
                        is_referential = True
                        break
                if is_referential:
                    break

        # Only remove if confirmed non-referential
        if not is_referential:
            indices_to_remove.add(token.i)

    return indices_to_remove


def remove_expletives_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """
    Remove expletive pronouns from a spaCy Doc (simple mode).

    This is the default registered function that uses simple dependency-based removal.
    For coreference resolution, use make_remove_expletives_with_coref() to create
    a configured function.

    Args:
        doc: spaCy Doc object containing the text to process

    Returns:
        Tuple of (ablated_text, num_removed)
    """
    indices_to_remove = find_and_confirm_expletives(doc, nlp_coref=None)

    if not indices_to_remove:
        return doc.text_with_ws, 0

    new_tokens = [tok.text_with_ws for i, tok in enumerate(doc) if i not in indices_to_remove]
    return "".join(new_tokens), len(indices_to_remove)


def make_remove_expletives_with_coref(nlp_coref: spacy.Language) -> Callable[[spacy.tokens.Doc], Tuple[str, int]]:
    """
    Create an expletive removal function with coreference resolution.

    This factory function creates a configured ablation function that uses
    coreference resolution to accurately distinguish between expletive and
    referential pronouns.

    Args:
        nlp_coref: spaCy pipeline with coreference resolution capability.
                   Typically a model like 'en_coreference_web_trf'.

    Returns:
        Ablation function with signature (doc) -> (ablated_text, num_removed)

    Example:
        >>> import spacy
        >>> nlp_coref = spacy.load("en_core_web_sm")  # or en_coreference_web_trf
        >>> ablate_fn = make_remove_expletives_with_coref(nlp_coref)
        >>> ablated_text, count = ablate_fn(doc)
    """
    def remove_expletives_with_coref(doc: spacy.tokens.Doc) -> Tuple[str, int]:
        """Remove expletives using coreference resolution."""
        indices_to_remove = find_and_confirm_expletives(doc, nlp_coref=nlp_coref)

        if not indices_to_remove:
            return doc.text_with_ws, 0

        new_tokens = [tok.text_with_ws for i, tok in enumerate(doc) if i not in indices_to_remove]
        return "".join(new_tokens), len(indices_to_remove)

    return remove_expletives_with_coref


def validate_expletive_removal(original: str, ablated: str, nlp) -> bool:
    """
    Validate that expletives were actually removed.

    Checks that the number of expletive tokens decreased from original to ablated text.

    Args:
        original: Original text before ablation
        ablated: Text after ablation
        nlp: spaCy pipeline for analysis

    Returns:
        True if expletives were reduced or none existed, False otherwise
    """
    # Process in smaller chunks to avoid spaCy's text length limit
    chunk_size = 500000

    def count_expletives(text):
        expletive_count = 0
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                doc = nlp(chunk)
                expletive_count += len([tok for tok in doc if tok.dep_ == 'expl'])
        return expletive_count

    original_expletives = count_expletives(original)
    ablated_expletives = count_expletives(ablated)

    if original_expletives > 0:
        return ablated_expletives < original_expletives
    else:
        # No expletives found - that's okay
        return True


# Register the simple (default) ablation with the registry
# For coreference resolution, use make_remove_expletives_with_coref() in your pipeline
AblationRegistry.register(
    "remove_expletives",
    remove_expletives_doc,
    validate_expletive_removal
)
