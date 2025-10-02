"""
Model creation module.

This module provides the main entry point for creating language models
using the multi-architecture factory pattern.

For backwards compatibility, this module re-exports the create_model_from_config
function as create_model.
"""

from .config import ExperimentConfig
from .architectures import create_model_from_config, BaseLanguageModel
from .architectures import _register_default_architectures


def create_model(config: ExperimentConfig, **kwargs) -> BaseLanguageModel:
    """
    Create a language model from configuration.

    This function creates a model using the multi-architecture factory.
    It automatically selects the appropriate architecture based on the
    'architecture' field in the model configuration.

    Args:
        config: ExperimentConfig object with model configuration
        **kwargs: Additional arguments passed to model constructor
                  (e.g., attn_implementation for Flash Attention)

    Returns:
        Instance of BaseLanguageModel (GPT2Model, BERTModel, LSTMModel, etc.)

    Raises:
        ValueError: If architecture field is missing or invalid

    Example:
        config = load_config("configs/experiment_gpt2.yaml")
        model = create_model(config)

    Note:
        The config must include an 'architecture' field in the model section.
        For example:
            model:
              architecture: "gpt2"
              transformer:
                layers: 12
                ...
    """
    # Register default architectures on first call
    _register_default_architectures()

    # Use the factory to create the model
    return create_model_from_config(config, **kwargs)