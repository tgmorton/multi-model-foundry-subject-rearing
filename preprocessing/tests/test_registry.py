"""
Unit tests for the AblationRegistry.

Tests registration, retrieval, and management of ablation functions.
"""

import pytest
from typing import Tuple
import spacy

from preprocessing.registry import AblationRegistry


# Test ablation functions
def dummy_ablation_1(doc) -> Tuple[str, int]:
    """Dummy ablation function for testing."""
    return "ablated", 1


def dummy_ablation_2(doc) -> Tuple[str, int]:
    """Another dummy ablation function."""
    return "modified", 2


def dummy_validator(original: str, ablated: str, nlp) -> bool:
    """Dummy validator function."""
    return len(ablated) < len(original)


class TestAblationRegistry:
    """Tests for AblationRegistry class."""

    def setup_method(self):
        """Clear registry before each test in this class."""
        from preprocessing.registry import AblationRegistry
        AblationRegistry.clear()

    def teardown_method(self):
        """Restore registry after each test in this class."""
        from preprocessing.registry import AblationRegistry
        # Re-register session-level ablations
        try:
            import importlib
            import preprocessing.ablations.remove_articles
            importlib.reload(preprocessing.ablations.remove_articles)
        except (ImportError, AttributeError):
            pass  # OK if not available

    def test_register_ablation_without_validator(self):
        """Can register ablation without validator."""
        AblationRegistry.register("test_ablation", dummy_ablation_1)

        assert AblationRegistry.is_registered("test_ablation")
        ablation_fn, validator_fn = AblationRegistry.get("test_ablation")
        assert ablation_fn == dummy_ablation_1
        assert validator_fn is None

    def test_register_ablation_with_validator(self):
        """Can register ablation with validator."""
        AblationRegistry.register("test_ablation", dummy_ablation_1, dummy_validator)

        ablation_fn, validator_fn = AblationRegistry.get("test_ablation")
        assert ablation_fn == dummy_ablation_1
        assert validator_fn == dummy_validator

    def test_register_duplicate_name_raises_error(self):
        """Registering duplicate name raises ValueError."""
        AblationRegistry.register("test_ablation", dummy_ablation_1)

        with pytest.raises(ValueError) as exc_info:
            AblationRegistry.register("test_ablation", dummy_ablation_2)

        assert "already registered" in str(exc_info.value)

    def test_get_unregistered_ablation_raises_error(self):
        """Getting unregistered ablation raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            AblationRegistry.get("nonexistent")

        assert "not found in registry" in str(exc_info.value)

    def test_get_includes_available_ablations_in_error(self):
        """Error message includes list of available ablations."""
        AblationRegistry.register("ablation_1", dummy_ablation_1)
        AblationRegistry.register("ablation_2", dummy_ablation_2)

        with pytest.raises(KeyError) as exc_info:
            AblationRegistry.get("nonexistent")

        error_msg = str(exc_info.value)
        assert "ablation_1" in error_msg
        assert "ablation_2" in error_msg

    def test_is_registered_returns_true_for_registered(self):
        """is_registered returns True for registered ablation."""
        AblationRegistry.register("test_ablation", dummy_ablation_1)

        assert AblationRegistry.is_registered("test_ablation") is True

    def test_is_registered_returns_false_for_unregistered(self):
        """is_registered returns False for unregistered ablation."""
        assert AblationRegistry.is_registered("nonexistent") is False

    def test_list_ablations_returns_sorted_list(self):
        """list_ablations returns sorted list of names."""
        AblationRegistry.register("zebra", dummy_ablation_1)
        AblationRegistry.register("apple", dummy_ablation_2)
        AblationRegistry.register("mango", dummy_ablation_1)

        ablations = AblationRegistry.list_ablations()

        assert ablations == ["apple", "mango", "zebra"]

    def test_list_ablations_empty_when_no_registrations(self):
        """list_ablations returns empty list when nothing registered."""
        ablations = AblationRegistry.list_ablations()
        assert ablations == []

    def test_unregister_removes_ablation(self):
        """unregister removes an ablation."""
        AblationRegistry.register("test_ablation", dummy_ablation_1, dummy_validator)
        AblationRegistry.unregister("test_ablation")

        assert not AblationRegistry.is_registered("test_ablation")

    def test_unregister_nonexistent_raises_error(self):
        """Unregistering nonexistent ablation raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            AblationRegistry.unregister("nonexistent")

        assert "not registered" in str(exc_info.value)

    def test_clear_removes_all_ablations(self):
        """clear removes all registered ablations."""
        AblationRegistry.register("ablation_1", dummy_ablation_1)
        AblationRegistry.register("ablation_2", dummy_ablation_2)

        AblationRegistry.clear()

        assert AblationRegistry.list_ablations() == []

    def test_ablation_function_callable(self):
        """Registered ablation function can be called."""
        AblationRegistry.register("test_ablation", dummy_ablation_1)

        ablation_fn, _ = AblationRegistry.get("test_ablation")
        result, count = ablation_fn(None)  # Doc doesn't matter for dummy

        assert result == "ablated"
        assert count == 1

    def test_validator_function_callable(self):
        """Registered validator function can be called."""
        AblationRegistry.register("test_ablation", dummy_ablation_1, dummy_validator)

        _, validator_fn = AblationRegistry.get("test_ablation")
        result = validator_fn("original text", "short", None)

        assert result is True
