"""
Unit tests for the StructuredLogger class.

Tests cover:
- Logger initialization with base context
- Structured JSON output format
- Log level methods (debug, info, warning, error, critical)
- Context merging and overriding
- Context management (update, clear)
- Exception logging with tracebacks
- Multiple logger independence
- Non-serializable value handling
"""

import json
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from model_foundry.logging_components import StructuredLogger


class TestStructuredLoggerInitialization:
    """Test StructuredLogger initialization and setup."""

    def test_creates_logger_with_base_context(self, tiny_config):
        """
        GIVEN: A valid experiment configuration
        WHEN: Creating a StructuredLogger instance
        THEN: Logger should have base context with experiment, git_hash, device
        """
        logger = StructuredLogger("test.logger", tiny_config)

        # Verify base context fields
        assert logger.context["experiment"] == tiny_config.experiment_name
        assert "git_hash" in logger.context
        assert "device" in logger.context
        assert isinstance(logger.logger, logging.Logger)
        assert logger.logger.name == "test.logger"

    def test_logger_name_matches_input(self, tiny_config):
        """
        GIVEN: A specific logger name
        WHEN: Creating StructuredLogger
        THEN: Internal logger should have that exact name
        """
        logger = StructuredLogger("model_foundry.custom.path", tiny_config)
        assert logger.logger.name == "model_foundry.custom.path"

    def test_base_context_immutable_across_instances(self, tiny_config):
        """
        GIVEN: Multiple StructuredLogger instances
        WHEN: Modifying context in one instance
        THEN: Other instances should not be affected
        """
        logger1 = StructuredLogger("logger1", tiny_config)
        logger2 = StructuredLogger("logger2", tiny_config)

        logger1.context["custom_field"] = "value1"

        # logger2's context should not have this field
        assert "custom_field" not in logger2.context


class TestStructuredLoggerOutput:
    """Test structured JSON output formatting."""

    def test_log_structured_creates_json_output(self, tiny_config, tmp_path):
        """
        GIVEN: A StructuredLogger with file handler
        WHEN: Logging a message with custom fields
        THEN: Output should be valid JSON with message and context
        """
        logger = StructuredLogger("test", tiny_config)
        log_file = tmp_path / "test.log"

        # Add file handler
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.DEBUG)
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.DEBUG)

        # Log a message
        logger.log_structured(logging.INFO, "Test message", custom_field="value123")
        handler.flush()

        # Read and parse
        log_content = log_file.read_text().strip()
        log_entry = json.loads(log_content)

        # Verify structure
        assert log_entry["message"] == "Test message"
        assert "context" in log_entry
        assert log_entry["context"]["experiment"] == tiny_config.experiment_name
        assert log_entry["context"]["custom_field"] == "value123"
        assert "git_hash" in log_entry["context"]

    def test_output_format_has_all_base_fields(self, tiny_config, tmp_path):
        """
        GIVEN: StructuredLogger
        WHEN: Logging any message
        THEN: Output must contain: message, context.experiment, context.git_hash, context.device
        """
        logger = StructuredLogger("test", tiny_config)
        log_file = tmp_path / "test.log"

        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.DEBUG)
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.DEBUG)

        logger.info("Test message")
        handler.flush()

        log_entry = json.loads(log_file.read_text().strip())

        # Required fields
        assert "message" in log_entry
        assert log_entry["message"] == "Test message"
        assert "context" in log_entry
        assert "experiment" in log_entry["context"]
        assert "git_hash" in log_entry["context"]
        assert "device" in log_entry["context"]


class TestStructuredLoggerLevels:
    """Test log level methods (debug, info, warning, error, critical)."""

    def test_info_level_logs_at_info(self, tiny_config):
        """
        GIVEN: StructuredLogger
        WHEN: Calling info() method
        THEN: Should log at INFO level (logging.INFO = 20)
        """
        logger = StructuredLogger("test", tiny_config)

        with patch.object(logger.logger, 'log') as mock_log:
            logger.info("Test message")

            mock_log.assert_called_once()
            assert mock_log.call_args[0][0] == logging.INFO

    def test_debug_level_logs_at_debug(self, tiny_config):
        """
        GIVEN: StructuredLogger
        WHEN: Calling debug() method
        THEN: Should log at DEBUG level (logging.DEBUG = 10)
        """
        logger = StructuredLogger("test", tiny_config)

        with patch.object(logger.logger, 'log') as mock_log:
            logger.debug("Test message")

            mock_log.assert_called_once()
            assert mock_log.call_args[0][0] == logging.DEBUG

    def test_warning_level_logs_at_warning(self, tiny_config):
        """
        GIVEN: StructuredLogger
        WHEN: Calling warning() method
        THEN: Should log at WARNING level (logging.WARNING = 30)
        """
        logger = StructuredLogger("test", tiny_config)

        with patch.object(logger.logger, 'log') as mock_log:
            logger.warning("Test message")

            mock_log.assert_called_once()
            assert mock_log.call_args[0][0] == logging.WARNING

    def test_error_level_logs_at_error(self, tiny_config):
        """
        GIVEN: StructuredLogger
        WHEN: Calling error() method
        THEN: Should log at ERROR level (logging.ERROR = 40)
        """
        logger = StructuredLogger("test", tiny_config)

        with patch.object(logger.logger, 'log') as mock_log:
            logger.error("Test message")

            mock_log.assert_called_once()
            assert mock_log.call_args[0][0] == logging.ERROR

    def test_critical_level_logs_at_critical(self, tiny_config):
        """
        GIVEN: StructuredLogger
        WHEN: Calling critical() method
        THEN: Should log at CRITICAL level (logging.CRITICAL = 50)
        """
        logger = StructuredLogger("test", tiny_config)

        with patch.object(logger.logger, 'log') as mock_log:
            logger.critical("Test message")

            mock_log.assert_called_once()
            assert mock_log.call_args[0][0] == logging.CRITICAL


class TestStructuredLoggerContext:
    """Test context management (merging, overriding, updating)."""

    def test_context_merges_with_base_context(self, tiny_config):
        """
        GIVEN: StructuredLogger with base context
        WHEN: Logging with additional context fields
        THEN: Final context should contain both base and custom fields
        """
        logger = StructuredLogger("test", tiny_config)

        with patch.object(logger.logger, 'log') as mock_log:
            logger.info("Test", step=100, loss=2.5, epoch=3)

            logged_message = mock_log.call_args[0][1]
            log_entry = json.loads(logged_message)

            # Should have both base and custom context
            assert "experiment" in log_entry["context"]
            assert "git_hash" in log_entry["context"]
            assert log_entry["context"]["step"] == 100
            assert log_entry["context"]["loss"] == 2.5
            assert log_entry["context"]["epoch"] == 3

    def test_custom_context_overrides_base_context(self, tiny_config):
        """
        GIVEN: StructuredLogger with base context
        WHEN: Logging with context field that matches base context key
        THEN: Custom value should override for that log message only
        """
        logger = StructuredLogger("test", tiny_config)
        original_experiment = logger.context["experiment"]

        with patch.object(logger.logger, 'log') as mock_log:
            logger.info("Test", experiment="override_experiment")

            logged_message = mock_log.call_args[0][1]
            log_entry = json.loads(logged_message)

            # Should be overridden in this message
            assert log_entry["context"]["experiment"] == "override_experiment"

            # But base context should remain unchanged
            assert logger.context["experiment"] == original_experiment

    def test_update_base_context(self, tiny_config):
        """
        GIVEN: StructuredLogger
        WHEN: Calling update_context() to add new base fields
        THEN: New fields should appear in all subsequent logs
        """
        logger = StructuredLogger("test", tiny_config)

        # Update base context
        logger.update_context(step=100, epoch=5)

        assert logger.context["step"] == 100
        assert logger.context["epoch"] == 5

        # Should appear in all logs
        with patch.object(logger.logger, 'log') as mock_log:
            logger.info("Test message")

            logged_message = mock_log.call_args[0][1]
            log_entry = json.loads(logged_message)

            assert log_entry["context"]["step"] == 100
            assert log_entry["context"]["epoch"] == 5

    def test_clear_context_field(self, tiny_config):
        """
        GIVEN: StructuredLogger with updated context
        WHEN: Calling clear_context_field() on a specific field
        THEN: That field should be removed from base context
        """
        logger = StructuredLogger("test", tiny_config)
        logger.update_context(step=100, temporary_field="value")

        assert "step" in logger.context
        assert "temporary_field" in logger.context

        # Clear one field
        logger.clear_context_field("temporary_field")

        assert "step" in logger.context
        assert "temporary_field" not in logger.context


class TestStructuredLoggerEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_non_serializable_context(self, tiny_config):
        """
        GIVEN: StructuredLogger
        WHEN: Logging with non-JSON-serializable context value
        THEN: Should convert to string representation without raising
        """
        logger = StructuredLogger("test", tiny_config)

        class NonSerializable:
            def __repr__(self):
                return "<NonSerializable object>"

        # Should not raise exception
        try:
            logger.info("Test", obj=NonSerializable())
        except (TypeError, ValueError):
            pytest.fail("Should handle non-serializable values gracefully")

    def test_log_exception_with_traceback(self, tiny_config, tmp_path):
        """
        GIVEN: An exception with traceback
        WHEN: Logging the exception with exc_info=True
        THEN: Log should contain traceback information
        """
        logger = StructuredLogger("test", tiny_config)
        log_file = tmp_path / "test.log"

        handler = logging.FileHandler(log_file)
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.ERROR)

        try:
            raise ValueError("Test error")
        except ValueError:
            # Log with exc_info to capture traceback
            logger.logger.error("Exception occurred", exc_info=True)

        handler.flush()
        log_content = log_file.read_text()

        # Should contain exception info
        assert "ValueError" in log_content
        assert "Test error" in log_content
        assert "Traceback" in log_content
