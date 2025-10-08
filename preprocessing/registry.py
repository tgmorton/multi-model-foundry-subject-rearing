"""
Ablation Function Registry

Centralized registration system for ablation functions and their validators.
This enables dynamic discovery and invocation of ablation transformations.
"""

from typing import Callable, Dict, Optional, Tuple, Any

# Type aliases for clarity
# Using Any for spaCy types to avoid import at module level
AblationFunction = Callable[[Any], Tuple[str, int]]  # Takes spacy.tokens.Doc
ValidationFunction = Callable[[str, str, Any], bool]  # Takes (str, str, spacy.Language)


class AblationRegistry:
    """
    Registry for ablation functions and their validators.

    This class maintains a mapping of ablation names to their implementation
    functions and optional validation functions. Ablation modules register
    themselves at import time.

    Example:
        >>> def remove_articles_doc(doc):
        ...     # Implementation
        ...     return ablated_text, num_removed
        >>>
        >>> AblationRegistry.register("remove_articles", remove_articles_doc)
        >>> ablation_fn, validator_fn = AblationRegistry.get("remove_articles")
    """

    _ablations: Dict[str, AblationFunction] = {}
    _validators: Dict[str, ValidationFunction] = {}

    @classmethod
    def register(
        cls,
        name: str,
        ablation_fn: AblationFunction,
        validation_fn: Optional[ValidationFunction] = None
    ) -> None:
        """
        Register an ablation function with optional validator.

        Args:
            name: Unique identifier for this ablation (e.g., "remove_articles")
            ablation_fn: Function that performs the ablation on a spaCy Doc
            validation_fn: Optional function to validate ablation occurred

        Raises:
            ValueError: If ablation name is already registered
        """
        if name in cls._ablations:
            raise ValueError(
                f"Ablation '{name}' is already registered. "
                "Use a unique name or unregister first."
            )

        cls._ablations[name] = ablation_fn
        if validation_fn:
            cls._validators[name] = validation_fn

    @classmethod
    def get(cls, name: str) -> Tuple[AblationFunction, Optional[ValidationFunction]]:
        """
        Retrieve a registered ablation function and its validator.

        Args:
            name: Name of the ablation to retrieve

        Returns:
            Tuple of (ablation_function, validation_function or None)

        Raises:
            KeyError: If ablation name is not registered
        """
        if name not in cls._ablations:
            available = ", ".join(cls._ablations.keys()) if cls._ablations else "none"
            raise KeyError(
                f"Ablation '{name}' not found in registry. "
                f"Available ablations: {available}"
            )

        return cls._ablations[name], cls._validators.get(name)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if an ablation is registered.

        Args:
            name: Name of the ablation to check

        Returns:
            True if registered, False otherwise
        """
        return name in cls._ablations

    @classmethod
    def list_ablations(cls) -> list[str]:
        """
        Get list of all registered ablation names.

        Returns:
            List of registered ablation names, sorted alphabetically
        """
        return sorted(cls._ablations.keys())

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister an ablation (mainly for testing).

        Args:
            name: Name of the ablation to unregister

        Raises:
            KeyError: If ablation name is not registered
        """
        if name not in cls._ablations:
            raise KeyError(f"Ablation '{name}' not registered")

        del cls._ablations[name]
        if name in cls._validators:
            del cls._validators[name]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered ablations (mainly for testing)."""
        cls._ablations.clear()
        cls._validators.clear()
