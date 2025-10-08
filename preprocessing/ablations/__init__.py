"""
Ablation Functions

This package contains individual ablation functions that transform text corpora.
Each module implements a specific linguistic ablation and registers itself
with the AblationRegistry.

Available ablations are automatically discovered and registered on import.
"""

# Import all ablation modules to trigger registration
from . import remove_articles
from . import remove_expletives
from . import impoverish_determiners
from . import lemmatize_verbs
from . import remove_subject_pronominals

__all__ = [
    "remove_articles",
    "remove_expletives",
    "impoverish_determiners",
    "lemmatize_verbs",
    "remove_subject_pronominals",
]
