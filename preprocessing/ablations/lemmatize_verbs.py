"""
Lemmatize all verbs to their infinitive form.

This ablation reduces all verbs to their base lemma form (e.g., "running" -> "run",
"went" -> "go"). This tests how models learn when verb morphology is impoverished.
"""

from typing import Tuple
import spacy
from preprocessing.registry import AblationRegistry


def lemmatize_verbs_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """
    Lemmatize all verbs to their infinitive form in a spaCy Doc.

    Args:
        doc: spaCy Doc object containing the text to process

    Returns:
        Tuple of (ablated_text, num_lemmatized)
    """
    modified_parts = []
    num_lemmatized = 0

    for token in doc:
        if token.pos_ == "VERB":
            modified_parts.append(token.lemma_ + token.whitespace_)
            num_lemmatized += 1
        else:
            modified_parts.append(token.text_with_ws)

    result = ''.join(modified_parts)
    return result, num_lemmatized


def validate_verb_lemmatization(original: str, ablated: str, nlp) -> bool:
    """
    Validate that verbs were actually lemmatized.

    Checks that verb forms changed between original and ablated text.

    Args:
        original: Original text before ablation
        ablated: Text after ablation
        nlp: spaCy pipeline for analysis

    Returns:
        True if verbs were found and lemmatized, False otherwise
    """
    original_doc = nlp(original)
    ablated_doc = nlp(ablated)

    original_verbs = [token.text for token in original_doc if token.pos_ == 'VERB']
    ablated_verbs = [token.text for token in ablated_doc if token.pos_ == 'VERB']

    if original_verbs:
        # Check if any verbs were lemmatized (different forms)
        original_verb_forms = set(original_verbs)
        ablated_verb_forms = set(ablated_verbs)
        lemmatized_count = len(original_verb_forms - ablated_verb_forms)
        return lemmatized_count > 0
    else:
        # No verbs found - that's okay
        return True


# Register the ablation with the registry
AblationRegistry.register(
    "lemmatize_verbs",
    lemmatize_verbs_doc,
    validate_verb_lemmatization
)
