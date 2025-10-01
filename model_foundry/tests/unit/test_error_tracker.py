"""
Unit tests for the ErrorTracker class.

Tests cover:
- Error logging to JSONL file
- Error count tracking by type
- Traceback capture
- Error summary generation
- Logging without context
- Counter reset
- Max errors limit
- Recent errors retrieval
"""

import json
import logging
from pathlib import Path
import pytest

from model_foundry.logging_components import ErrorTracker


class TestErrorTrackerBasics:
    """Test basic error tracking functionality."""

    def test_log_error_writes_to_file(self, tmp_path):
        """
        GIVEN: ErrorTracker instance
        WHEN: Logging an exception
        THEN: Should write to errors.jsonl with error details
        """
        logger = logging.getLogger("test")
        tracker = ErrorTracker(logger, tmp_path)

        try:
            raise ValueError("Test error message")
        except ValueError as e:
            tracker.log_error(e, context={"step": 100, "epoch": 5})

        error_log = tmp_path / "errors.jsonl"
        assert error_log.exists()

        with open(error_log, 'r') as f:
            entry = json.loads(f.readline())

        # Verify structure
        assert entry["error_type"] == "ValueError"
        assert entry["error_message"] == "Test error message"
        assert "traceback" in entry
        assert entry["context"]["step"] == 100
        assert entry["context"]["epoch"] == 5
        assert "timestamp" in entry

    def test_log_error_increments_counter(self, tmp_path):
        """
        GIVEN: ErrorTracker
        WHEN: Logging multiple errors of different types
        THEN: Should track counts by error type
        """
        logger = logging.getLogger("test")
        tracker = ErrorTracker(logger, tmp_path)

        # Log 3 ValueErrors
        for _ in range(3):
            try:
                raise ValueError("Test")
            except ValueError as e:
                tracker.log_error(e)

        # Log 2 TypeErrors
        for _ in range(2):
            try:
                raise TypeError("Test")
            except TypeError as e:
                tracker.log_error(e)

        summary = tracker.get_error_summary()

        assert summary["ValueError"] == 3
        assert summary["TypeError"] == 2

    def test_log_error_includes_traceback(self, tmp_path):
        """
        GIVEN: Exception raised in nested function
        WHEN: Logging the error
        THEN: Traceback should show full call stack
        """
        logger = logging.getLogger("test")
        tracker = ErrorTracker(logger, tmp_path)

        def nested_function():
            raise RuntimeError("Nested error")

        def outer_function():
            nested_function()

        try:
            outer_function()
        except RuntimeError as e:
            tracker.log_error(e)

        with open(tmp_path / "errors.jsonl", 'r') as f:
            entry = json.loads(f.readline())

        # Should contain function names in traceback
        assert "nested_function" in entry["traceback"]
        assert "outer_function" in entry["traceback"]
        assert "RuntimeError" in entry["traceback"]


class TestErrorTrackerAggregation:
    """Test error aggregation and summary."""

    def test_get_error_summary(self, tmp_path):
        """
        GIVEN: Multiple errors of various types
        WHEN: Getting error summary
        THEN: Should return dict with counts for each type
        """
        logger = logging.getLogger("test")
        tracker = ErrorTracker(logger, tmp_path)

        error_types = [ValueError, TypeError, ValueError, RuntimeError, ValueError]

        for error_cls in error_types:
            try:
                raise error_cls("Test")
            except error_cls as e:
                tracker.log_error(e)

        summary = tracker.get_error_summary()

        assert summary == {
            "ValueError": 3,
            "TypeError": 1,
            "RuntimeError": 1
        }

    def test_log_error_with_no_context(self, tmp_path):
        """
        GIVEN: Error without additional context
        WHEN: Logging error
        THEN: Context field should be empty dict
        """
        logger = logging.getLogger("test")
        tracker = ErrorTracker(logger, tmp_path)

        try:
            raise ValueError("Test")
        except ValueError as e:
            tracker.log_error(e)  # No context provided

        with open(tmp_path / "errors.jsonl", 'r') as f:
            entry = json.loads(f.readline())

        assert entry["context"] == {}


class TestErrorTrackerManagement:
    """Test error tracker management functionality."""

    def test_reset_error_counts(self, tmp_path):
        """
        GIVEN: ErrorTracker with accumulated errors
        WHEN: Calling reset_counters()
        THEN: Error counts should be cleared
        """
        logger = logging.getLogger("test")
        tracker = ErrorTracker(logger, tmp_path)

        try:
            raise ValueError("Test")
        except ValueError as e:
            tracker.log_error(e)

        assert tracker.get_error_summary()["ValueError"] == 1

        tracker.reset_counters()

        assert len(tracker.get_error_summary()) == 0

    def test_max_errors_limit(self, tmp_path):
        """
        GIVEN: ErrorTracker with max_errors limit
        WHEN: Logging more errors than limit
        THEN: Counter should still be accurate
        """
        logger = logging.getLogger("test")
        tracker = ErrorTracker(logger, tmp_path, max_errors=10)

        # Log 15 errors
        for i in range(15):
            try:
                raise ValueError(f"Error {i}")
            except ValueError as e:
                tracker.log_error(e)

        # Counter should still be accurate
        assert tracker.get_error_summary()["ValueError"] == 15

        # File should have all 15 entries
        with open(tmp_path / "errors.jsonl", 'r') as f:
            lines = f.readlines()

        assert len(lines) == 15

    def test_get_recent_errors(self, tmp_path):
        """
        GIVEN: Multiple errors logged
        WHEN: Retrieving recent errors
        THEN: Should return N most recent in reverse chronological order
        """
        logger = logging.getLogger("test")
        tracker = ErrorTracker(logger, tmp_path)

        # Log 10 errors with different indices
        for i in range(10):
            try:
                raise ValueError(f"Error {i}")
            except ValueError as e:
                tracker.log_error(e, context={"index": i})

        recent = tracker.get_recent_errors(n=3)

        assert len(recent) == 3

        # Should be most recent (9, 8, 7) in that order
        assert recent[0]["context"]["index"] == 9
        assert recent[1]["context"]["index"] == 8
        assert recent[2]["context"]["index"] == 7
