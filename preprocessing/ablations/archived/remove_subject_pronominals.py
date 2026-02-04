"""
Remove subject pronominals (pronouns functioning as subjects).

This ablation removes all pronouns that function as nominal subjects (nsubj dependency).
This tests how models learn subject-drop patterns when explicit subject pronouns
are removed from the training data.
"""

from typing import Tuple
import spacy
from preprocessing.registry import AblationRegistry


def remove_subject_pronominals_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """
    Remove all pronouns functioning as nominal subjects from a spaCy Doc.

    Args:
        doc: spaCy Doc object containing the text to process

    Returns:
        Tuple of (ablated_text, num_removed)
    """
    modified_parts = []
    num_removed = 0

    for token in doc:
        # Check if token is a pronoun functioning as a nominal subject
        is_subj_pronoun = token.pos_ == 'PRON' and token.dep_ == 'nsubj'

        if not is_subj_pronoun:
            # Preserve original spacing
            modified_parts.append(token.text_with_ws)
        else:
            num_removed += 1

    result = ''.join(modified_parts)
    return result, num_removed


def validate_subject_pronoun_removal(original: str, ablated: str, nlp) -> bool:
    """
    Validate that subject pronouns were actually removed.

    Checks that the number of subject pronouns decreased from original to ablated text.

    Args:
        original: Original text before ablation
        ablated: Text after ablation
        nlp: spaCy pipeline for analysis

    Returns:
        True if subject pronouns were reduced or none existed, False otherwise
    """
    original_doc = nlp(original)
    ablated_doc = nlp(ablated)

    original_subj_pronouns = [
        token.text for token in original_doc
        if token.pos_ == 'PRON' and token.dep_ == 'nsubj'
    ]
    ablated_subj_pronouns = [
        token.text for token in ablated_doc
        if token.pos_ == 'PRON' and token.dep_ == 'nsubj'
    ]

    if original_subj_pronouns:
        return len(ablated_subj_pronouns) < len(original_subj_pronouns)
    else:
        # No subject pronouns found - that's okay
        return True


# Register the ablation with the registry
AblationRegistry.register(
    "remove_subject_pronominals",
    remove_subject_pronominals_doc,
    validate_subject_pronoun_removal
)
