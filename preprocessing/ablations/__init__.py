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
from . import identity
from . import lemmatize_verbs
from . import remove_expletive_sentences
from . import impoverish_case
from . import enrich_verbal_morphology
from . import remove_subject_pronouns_graded
from . import stacked_graded  # must import after the interventions it composes

__all__ = [
    "identity",
    "lemmatize_verbs",
    "remove_expletive_sentences",
    "impoverish_case",
    "enrich_verbal_morphology",
    "remove_subject_pronouns_graded",
    "stacked_graded",
]
