"""
Remove expletives (dummy pronouns) from text.

This ablation identifies and removes expletive pronouns (e.g., "it" in "it is raining")
that serve no referential function. Uses coreference resolution to distinguish between
referential and non-referential pronouns.
"""

from typing import Tuple
import spacy
from preprocessing.registry import AblationRegistry


def find_and_confirm_expletives(doc: spacy.tokens.Doc, nlp_coref) -> set:
    """
    Find potential dummy pronouns and confirm using coreference resolution.

    Implements a two-step procedure:
    1. Identifies tokens with dependency label 'expl' (expletive)
    2. Confirms they are non-referential using coreference resolution

    Args:
        doc: spaCy Doc object to analyze
        nlp_coref: spaCy pipeline with coreference resolution

    Returns:
        Set of token indices to remove
    """
    indices_to_remove = set()
    potential_dummies = [
        tok for tok in doc if tok.dep_ == 'expl' and tok.head.pos_ == 'VERB'
    ]

    for token in potential_dummies:
        current_sent = token.sent
        prev_sent = None
        if current_sent.start > 0:
            if token.doc is doc:
                prev_token = doc[current_sent.start - 1]
                prev_sent = prev_token.sent

        context_text = prev_sent.text + " " + current_sent.text if prev_sent else current_sent.text

        coref_doc = nlp_coref(context_text)

        is_referential = False
        if 'coref' in coref_doc.spans:
            for i, cluster in enumerate(coref_doc.spans['coref']):
                for mention in cluster:
                    if token.text.lower() == mention.text.lower():
                        is_referential = True
                        break
                if is_referential:
                    break

        if not is_referential:
            indices_to_remove.add(token.i)

    return indices_to_remove


def remove_expletives_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """
    Remove expletive pronouns from a spaCy Doc.

    Note: This is a simplified version that just removes tokens with 'expl' dependency.
    For full coreference resolution, use the complete pipeline with nlp_coref.

    Args:
        doc: spaCy Doc object containing the text to process

    Returns:
        Tuple of (ablated_text, num_removed)
    """
    # For the basic version, just remove tokens marked as expletives
    indices_to_remove = {tok.i for tok in doc if tok.dep_ == 'expl' and tok.head.pos_ == 'VERB'}

    if not indices_to_remove:
        return doc.text_with_ws, 0

    new_tokens = [tok.text_with_ws for i, tok in enumerate(doc) if i not in indices_to_remove]
    return "".join(new_tokens), len(indices_to_remove)


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


# Register the ablation with the registry
AblationRegistry.register(
    "remove_expletives",
    remove_expletives_doc,
    validate_expletive_removal
)
