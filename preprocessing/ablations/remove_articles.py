"""
Remove Articles Ablation

Removes all articles ('a', 'an', 'the') from the corpus.
This ablation tests how models learn when determiner morphology is impoverished.

Procedure: RemoveArticles(text)
1. Load spaCy NLP model with POS tagger
2. For each token in doc:
3.   is_article = token.pos_ == 'DET' and token.lower_ in ['a', 'an', 'the']
4.   If not is_article, keep token
5. Return modified text and count of articles removed

Based on original implementation from preprocessing/remove_articles.py
"""

from typing import Tuple
import spacy

from preprocessing.registry import AblationRegistry


def remove_articles_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """
    Remove all articles ('a', 'an', 'the') from a spaCy Doc.

    This function identifies determiners that are articles based on POS tags
    and their lowercase form, then removes them from the text while preserving
    whitespace structure.

    Args:
        doc: spaCy Doc object to process

    Returns:
        Tuple of (ablated_text, num_articles_removed)
            - ablated_text: Text with articles removed
            - num_articles_removed: Count of articles that were removed
    """
    modified_parts = []
    num_removed = 0

    for token in doc:
        # Check if token is an article (determiner with specific forms)
        is_article = token.pos_ == 'DET' and token.lower_ in ['a', 'an', 'the']

        if not is_article:
            # Keep non-article tokens with their whitespace
            modified_parts.append(token.text_with_ws)
        else:
            # Count removed articles
            num_removed += 1

    result = ''.join(modified_parts)
    return result, num_removed


def validate_article_removal(
    original_text: str,
    ablated_text: str,
    nlp: spacy.Language
) -> bool:
    """
    Validate that articles were actually removed from the text.

    This validator processes both the original and ablated text to count
    articles and verify that the ablation reduced the article count.

    Args:
        original_text: Original text before ablation
        ablated_text: Text after ablation
        nlp: spaCy Language model for processing

    Returns:
        True if articles were found and removed (or no articles existed),
        False if ablation failed to remove articles
    """
    original_doc = nlp(original_text)
    ablated_doc = nlp(ablated_text)

    # Count articles in original
    original_articles = [
        token.text for token in original_doc
        if token.pos_ == 'DET' and token.lower_ in ['a', 'an', 'the']
    ]

    # Count articles in ablated
    ablated_articles = [
        token.text for token in ablated_doc
        if token.pos_ == 'DET' and token.lower_ in ['a', 'an', 'the']
    ]

    if original_articles:
        # If there were articles, we should have removed some
        return len(ablated_articles) < len(original_articles)
    else:
        # If no articles existed, ablation is trivially valid
        return True


# Register this ablation with the registry
# This happens automatically when the module is imported
AblationRegistry.register(
    "remove_articles",
    remove_articles_doc,
    validate_article_removal
)
