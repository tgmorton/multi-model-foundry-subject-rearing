"""
Sentence-level annotators for corpus descriptive analysis.

Annotators produce per-sentence annotation records that are written to
Parquet files, enabling downstream queries and data attribution.
"""

from .base import BaseSentenceAnnotator, CompositeAnnotator
from .clause_structure import ClauseStructureAnnotator
from .that_trace import ThatTraceAnnotator
from .expletive import ExpletiveAnnotator
from .pronoun import PronounAnnotator
from .negation import NegationAnnotator
from .wh_extraction import WhExtractionAnnotator
from .relative_clause import RelativeClauseAnnotator
from .verb import VerbAnnotator
from .argument_structure import ArgumentStructureAnnotator
from .topic import TopicAnnotator
from .complexity import ComplexityAnnotator
from .fragment import FragmentAnnotator


# Registry of all available annotators
ANNOTATOR_REGISTRY = {
    "clause_structure": ClauseStructureAnnotator,
    "that_trace": ThatTraceAnnotator,
    "expletive": ExpletiveAnnotator,
    "pronoun": PronounAnnotator,
    "negation": NegationAnnotator,
    "wh_extraction": WhExtractionAnnotator,
    "relative_clause": RelativeClauseAnnotator,
    "verb": VerbAnnotator,
    "argument_structure": ArgumentStructureAnnotator,
    "topic": TopicAnnotator,
    "complexity": ComplexityAnnotator,
    "fragment": FragmentAnnotator,
}


def get_default_annotators(language: str = "en") -> list[BaseSentenceAnnotator]:
    """
    Get a list of all default annotators.

    Args:
        language: Language code ("en" for English, "it" for Italian).
            Affects ExpletiveAnnotator, ThatTraceAnnotator,
            WhExtractionAnnotator, and RelativeClauseAnnotator.

    Returns all annotators except fragment (which overlaps with complexity).
    """
    return [
        ClauseStructureAnnotator(language=language),
        ThatTraceAnnotator(language=language),
        ExpletiveAnnotator(language=language),
        PronounAnnotator(),
        NegationAnnotator(),
        WhExtractionAnnotator(language=language),
        RelativeClauseAnnotator(language=language),
        VerbAnnotator(),
        ArgumentStructureAnnotator(language=language),
        TopicAnnotator(),
        ComplexityAnnotator(language=language),
    ]


def get_annotator(name: str, language: str = "en") -> BaseSentenceAnnotator:
    """
    Get an annotator instance by name.

    Args:
        name: Annotator name (e.g., "clause_structure", "that_trace")
        language: Language code ("en" for English, "it" for Italian).
            Affects language-aware annotators (expletive, that_trace,
            wh_extraction, relative_clause, fragment, clause_structure,
            argument_structure, complexity).

    Returns:
        Instance of the specified annotator

    Raises:
        ValueError: If annotator name is not recognized
    """
    if name not in ANNOTATOR_REGISTRY:
        raise ValueError(
            f"Unknown annotator: {name}. "
            f"Available: {list(ANNOTATOR_REGISTRY.keys())}"
        )
    # Pass language to annotators that need it
    language_aware = {
        "expletive", "that_trace", "wh_extraction", "relative_clause", "fragment",
        "clause_structure", "argument_structure", "complexity",
    }
    if name in language_aware:
        return ANNOTATOR_REGISTRY[name](language=language)
    return ANNOTATOR_REGISTRY[name]()


__all__ = [
    # Base classes
    "BaseSentenceAnnotator",
    "CompositeAnnotator",
    # Annotators
    "ClauseStructureAnnotator",
    "ThatTraceAnnotator",
    "ExpletiveAnnotator",
    "PronounAnnotator",
    "NegationAnnotator",
    "WhExtractionAnnotator",
    "RelativeClauseAnnotator",
    "VerbAnnotator",
    "ArgumentStructureAnnotator",
    "TopicAnnotator",
    "ComplexityAnnotator",
    "FragmentAnnotator",
    # Registry
    "ANNOTATOR_REGISTRY",
    # Helpers
    "get_default_annotators",
    "get_annotator",
]
