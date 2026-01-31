"""
Base class for corpus descriptive analyzers.
"""

from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, Optional

import spacy


class BaseAnalyzer(ABC):
    """
    Abstract base for single-pass corpus analyzers.

    Each analyzer receives every Doc exactly once via process_doc(),
    accumulates internal counters, and produces results keyed by genre.
    """

    def __init__(self):
        self._total_tokens = Counter()  # genre -> token count

    @abstractmethod
    def process_doc(
        self,
        doc: spacy.tokens.Doc,
        genre: str,
        speaker: Optional[str] = None,
    ) -> None:
        """Process a single spaCy Doc and update internal counters."""
        ...

    @abstractmethod
    def get_results(self) -> Dict[str, Any]:
        """
        Return accumulated results.

        Returns dict with at least:
          {"overall": {...}, "by_genre": {genre: {...}, ...}}
        """
        ...

    def merge(self, other: "BaseAnalyzer") -> None:
        """Merge counters from another analyzer instance of the same type."""
        if type(self) is not type(other):
            raise TypeError(
                f"Cannot merge {type(other).__name__} into {type(self).__name__}"
            )
        for key, count in other._total_tokens.items():
            self._total_tokens[key] += count

    def to_checkpoint(self) -> Dict[str, Any]:
        """Serialize state for checkpointing."""
        return {"_total_tokens": dict(self._total_tokens)}

    def from_checkpoint(self, data: Dict[str, Any]) -> None:
        """Restore state from a checkpoint."""
        self._total_tokens = Counter(data.get("_total_tokens", {}))

    @property
    def name(self) -> str:
        return type(self).__name__

    # ----- helpers -----

    @staticmethod
    def _counter_to_json(counter: Counter) -> Dict[str, int]:
        """Convert Counter to JSON-safe dict (stringify tuple keys)."""
        return {str(k): v for k, v in counter.items()}

    @staticmethod
    def _sum_counters(counter: Counter) -> Dict[str, int]:
        """Convert Counter to plain dict for JSON serialization."""
        return dict(counter)

    @staticmethod
    def _merge_counter_dicts(
        target: Dict[str, Counter], source: Dict[str, Counter]
    ) -> None:
        """Merge a dict-of-Counters from source into target in place."""
        for key, counter in source.items():
            if key in target:
                target[key] += counter
            else:
                target[key] = Counter(counter)
