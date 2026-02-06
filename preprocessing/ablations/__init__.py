"""
Ablation Functions

This package contains individual ablation functions that transform text corpora.
Each module implements a specific linguistic ablation and registers itself
with the AblationRegistry.

Available ablations are automatically discovered and registered on import.

Archived (inactive) ablations live in ``preprocessing/ablations/archived/``
and are NOT imported here.
"""

# Import all active ablation modules to trigger registration
from . import lemmatize_verbs
from . import remove_expletive_sentences
from . import impoverish_case
from . import enrich_verbal_morphology

__all__ = [
    "lemmatize_verbs",
    "remove_expletive_sentences",
    "impoverish_case",
    "enrich_verbal_morphology",
]
