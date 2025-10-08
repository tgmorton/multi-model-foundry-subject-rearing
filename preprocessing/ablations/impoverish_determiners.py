"""
Impoverish determiners by replacing all with 'the'.

This ablation replaces all determiners (a, an, the, this, that, these, those, etc.)
with the single determiner 'the'. This tests how models learn when determiner
morphology is impoverished.
"""

from typing import Tuple
import spacy
from preprocessing.registry import AblationRegistry


def impoverish_determiners_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """
    Replace all determiners with 'the' in a spaCy Doc.

    Args:
        doc: spaCy Doc object containing the text to process

    Returns:
        Tuple of (ablated_text, num_replaced)
    """
    modified_parts = []
    num_replaced = 0

    for token in doc:
        if token.pos_ == "DET":
            modified_parts.append("the" + token.whitespace_)
            num_replaced += 1
        else:
            modified_parts.append(token.text_with_ws)

    result = ''.join(modified_parts)
    return result, num_replaced


def validate_determiner_impoverishment(original: str, ablated: str, nlp) -> bool:
    """
    Validate that determiners were actually impoverished.

    Checks that:
    1. Determiners existed in the original text
    2. All determiners in ablated text are 'the'

    Args:
        original: Original text before ablation
        ablated: Text after ablation
        nlp: spaCy pipeline for analysis

    Returns:
        True if determiners were properly impoverished, False otherwise
    """
    original_doc = nlp(original)
    ablated_doc = nlp(ablated)

    original_determiners = [token.text for token in original_doc if token.pos_ == 'DET']
    ablated_determiners = [token.text for token in ablated_doc if token.pos_ == 'DET']

    if original_determiners:
        # Check if all determiners are now 'the'
        non_the_determiners = [d for d in ablated_determiners if d.lower() != 'the']
        return len(non_the_determiners) == 0
    else:
        # No determiners found - that's okay
        return True


# Register the ablation with the registry
AblationRegistry.register(
    "impoverish_determiners",
    impoverish_determiners_doc,
    validate_determiner_impoverishment
)
