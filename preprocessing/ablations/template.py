"""
[Ablation Name] - [Brief One-Line Description]

[Detailed description of what this ablation does, why it's useful,
and what linguistic phenomenon it tests.]

Example:
    >>> import spacy
    >>> nlp = spacy.load("en_core_web_sm")
    >>> doc = nlp("Your example text here")
    >>> ablated_text, count = my_ablation_doc(doc)
    >>> print(f"Modified: {count} items")
"""

from typing import Tuple
import spacy
from preprocessing.registry import AblationRegistry


def my_ablation_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """
    [One-line description of the transformation this function performs.]

    [More detailed explanation if needed, including:
    - What tokens are targeted
    - How they are modified (removed/replaced/changed)
    - Any special cases handled]

    Args:
        doc: spaCy Doc object containing the text to process

    Returns:
        Tuple of (ablated_text, num_modifications)
        - ablated_text: The transformed text with modifications applied
        - num_modifications: Count of items modified (removed/replaced/changed)

    Example:
        >>> import spacy
        >>> nlp = spacy.load("en_core_web_sm")
        >>> doc = nlp("Example text")
        >>> ablated, count = my_ablation_doc(doc)
    """
    modified_parts = []
    num_modifications = 0

    for token in doc:
        # TODO: Add your ablation logic here
        #
        # Common patterns:
        #
        # 1. Remove tokens:
        #    if condition:
        #        num_modifications += 1
        #        # Don't append to modified_parts
        #    else:
        #        modified_parts.append(token.text_with_ws)
        #
        # 2. Replace tokens:
        #    if condition:
        #        modified_parts.append("replacement" + token.whitespace_)
        #        num_modifications += 1
        #    else:
        #        modified_parts.append(token.text_with_ws)
        #
        # 3. Modify tokens:
        #    if condition:
        #        modified_parts.append(token.lemma_ + token.whitespace_)
        #        num_modifications += 1
        #    else:
        #        modified_parts.append(token.text_with_ws)

        # Example: Keep all tokens unchanged (replace with your logic)
        modified_parts.append(token.text_with_ws)

    return ''.join(modified_parts), num_modifications


def validate_my_ablation(original: str, ablated: str, nlp) -> bool:
    """
    Validate that the ablation occurred as expected.

    [Explain what this function checks to ensure the ablation worked correctly.]

    Args:
        original: Original text before ablation
        ablated: Text after ablation
        nlp: spaCy pipeline for analysis

    Returns:
        True if ablation was successful or no target items existed,
        False if ablation failed (target items still present unchanged)

    Example:
        >>> nlp = spacy.load("en_core_web_sm")
        >>> original = "Original text"
        >>> ablated = "Modified text"
        >>> is_valid = validate_my_ablation(original, ablated, nlp)
    """
    # Process both texts
    original_doc = nlp(original)
    ablated_doc = nlp(ablated)

    # TODO: Add your validation logic here
    #
    # Common pattern:
    # 1. Count target items in original
    # 2. Count target items in ablated
    # 3. Return True if ablated has fewer (or none if original had none)
    #
    # Example:
    # original_count = sum(1 for token in original_doc if <condition>)
    # ablated_count = sum(1 for token in ablated_doc if <condition>)
    #
    # if original_count > 0:
    #     return ablated_count < original_count
    # else:
    #     return True  # No target items existed - that's okay

    # Placeholder: always return True (replace with your validation)
    return True


# Register the ablation with the registry
# The name should be:
# - Lowercase
# - Underscore-separated
# - Descriptive (e.g., "remove_adjectives", "lowercase_text")
AblationRegistry.register(
    "my_ablation",  # TODO: Change this to your ablation name
    my_ablation_doc,
    validate_my_ablation
)
