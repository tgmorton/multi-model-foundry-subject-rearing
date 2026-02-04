"""
Base class for sentence-level annotators.

Unlike analyzers (which aggregate counts), annotators produce per-sentence
annotation records that are written to Parquet for downstream queries.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import spacy


class BaseSentenceAnnotator(ABC):
    """
    Abstract base for sentence-level annotators.

    Each annotator receives a spaCy Span (sentence) and returns a dict
    of annotations for that sentence. Annotations are accumulated into
    Parquet files rather than aggregated Counters.
    """

    # Subclasses should define the annotation fields they produce
    # This is used for schema validation and documentation
    output_fields: Dict[str, str] = {}  # field_name -> description

    @property
    def name(self) -> str:
        """Return the annotator name (used as layer name)."""
        return type(self).__name__

    @abstractmethod
    def annotate_sentence(
        self,
        sent: spacy.tokens.Span,
        genre: str,
        speaker: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Annotate a single sentence.

        Args:
            sent: spaCy Span representing the sentence
            genre: Corpus genre (e.g., "CHILDES", "BNC")
            speaker: Speaker identifier (e.g., "CHI", "MOT") if available
            metadata: Additional metadata dict (e.g., role, file info)

        Returns:
            Dict with annotation fields. Keys should match output_fields.
            Lists and nested dicts are supported for complex annotations.
        """
        ...

    def get_sentence_flags(
        self,
        annotations: Dict[str, Any],
    ) -> Dict[str, bool]:
        """
        Compute boolean flags from annotations for fast filtering.

        Override in subclasses to add sentence-level flags like
        has_null_subject, has_that_trace_violation, etc.

        Args:
            annotations: The annotation dict returned by annotate_sentence

        Returns:
            Dict of boolean flags derived from the annotations
        """
        return {}


class CompositeAnnotator(BaseSentenceAnnotator):
    """
    Combines multiple annotators into a single annotator.

    Useful for running all annotators in a single pass and merging results.
    """

    def __init__(self, annotators: list[BaseSentenceAnnotator]):
        self.annotators = annotators
        # Merge output_fields from all annotators
        self.output_fields = {}
        for ann in annotators:
            self.output_fields.update(ann.output_fields)

    @property
    def name(self) -> str:
        return "CompositeAnnotator"

    def annotate_sentence(
        self,
        sent: spacy.tokens.Span,
        genre: str,
        speaker: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run all annotators and merge results."""
        result = {}
        for annotator in self.annotators:
            annotations = annotator.annotate_sentence(sent, genre, speaker, metadata)
            result.update(annotations)
            # Also compute flags
            flags = annotator.get_sentence_flags(annotations)
            result.update(flags)
        return result
