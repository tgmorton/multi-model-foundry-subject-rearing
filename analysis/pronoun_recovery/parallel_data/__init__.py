"""
Europarl parallel alignment pipeline for pronoun recovery training data.

Uses awesome-align on EN-IT sentence pairs to extract high-quality
pronoun labels by mapping English subject pronouns to Italian
null-subject verbs.
"""

from .generator import EuroparlAlignmentGenerator

__all__ = ["EuroparlAlignmentGenerator"]
