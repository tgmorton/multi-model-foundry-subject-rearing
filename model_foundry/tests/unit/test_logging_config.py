"""
Unit tests for the LoggingConfig dataclass.

Tests cover:
- Default configuration values
- Custom configuration values
- Log level validation
- Positive integer validation
- Integration with ExperimentConfig
"""

import pytest
from pydantic import ValidationError

from model_foundry.config import LoggingConfig, ExperimentConfig


class TestLoggingConfigDefaults:
    """Test default configuration values."""

    def test_default_values(self):
        """
        GIVEN: LoggingConfig with no arguments
        WHEN: Creating instance
        THEN: Should have sensible defaults
        """
        config = LoggingConfig()

        assert config.console_level == "INFO"
        assert config.file_level == "DEBUG"
        assert config.dir == "logs"
        assert config.use_structured_logging is True
        assert config.use_wandb is False
        assert config.wandb_project is None
        assert config.max_log_files == 10
        assert config.max_log_size_mb == 100
        assert config.log_metrics_every_n_steps == 10
        assert config.log_detailed_metrics_every_n_steps == 100
        assert config.profile_performance is False
        assert config.log_memory_every_n_steps == 100
        assert config.max_errors_to_track == 1000


class TestLoggingConfigCustomization:
    """Test custom configuration values."""

    def test_custom_values(self):
        """
        GIVEN: Custom configuration values
        WHEN: Creating LoggingConfig
        THEN: Should use custom values
        """
        config = LoggingConfig(
            console_level="WARNING",
            file_level="INFO",
            use_structured_logging=False,
            max_log_files=5,
            log_metrics_every_n_steps=50
        )

        assert config.console_level == "WARNING"
        assert config.file_level == "INFO"
        assert config.use_structured_logging is False
        assert config.max_log_files == 5
        assert config.log_metrics_every_n_steps == 50


class TestLoggingConfigValidation:
    """Test configuration validation."""

    def test_validates_log_levels(self):
        """
        GIVEN: Valid and invalid log level strings
        WHEN: Creating LoggingConfig
        THEN: Valid levels should work, invalid should raise ValueError
        """
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in valid_levels:
            config = LoggingConfig(console_level=level, file_level=level)
            assert config.console_level == level
            assert config.file_level == level

        # Invalid level should raise
        with pytest.raises(ValueError, match="console_level must be one of"):
            LoggingConfig(console_level="INVALID_LEVEL")

    def test_validates_positive_integers(self):
        """
        GIVEN: Negative or zero integer values
        WHEN: Creating LoggingConfig
        THEN: Should raise ValidationError
        """
        # max_log_files must be positive
        with pytest.raises(ValidationError):
            LoggingConfig(max_log_files=-1)

        with pytest.raises(ValidationError):
            LoggingConfig(max_log_files=0)

        # max_log_size_mb must be positive
        with pytest.raises(ValidationError):
            LoggingConfig(max_log_size_mb=0)

        # log_metrics_every_n_steps must be positive
        with pytest.raises(ValidationError):
            LoggingConfig(log_metrics_every_n_steps=-10)


class TestLoggingConfigIntegration:
    """Test integration with ExperimentConfig."""

    def test_integrates_with_experiment_config(self, tiny_config):
        """
        GIVEN: ExperimentConfig with LoggingConfig
        WHEN: Creating full experiment config
        THEN: Should validate and integrate successfully
        """
        # tiny_config already has logging config, just verify it exists
        assert hasattr(tiny_config, 'logging')
        assert isinstance(tiny_config.logging, LoggingConfig)

    def test_custom_logging_config_in_experiment(self, tiny_config):
        """
        GIVEN: ExperimentConfig with custom LoggingConfig
        WHEN: Setting custom logging values
        THEN: Should accept and use custom values
        """
        # Convert tiny_config to dict and modify logging config
        config_dict = tiny_config.dict()
        config_dict['logging'] = {
            'console_level': 'DEBUG',
            'file_level': 'DEBUG',
            'use_structured_logging': True,
            'log_metrics_every_n_steps': 5,
            'profile_performance': True
        }

        # Should validate successfully
        full_config = ExperimentConfig(**config_dict)

        assert full_config.logging.console_level == "DEBUG"
        assert full_config.logging.log_metrics_every_n_steps == 5
        assert full_config.logging.profile_performance is True
