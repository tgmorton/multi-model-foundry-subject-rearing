"""
Multi-architecture model factory and registry.

This module provides the registry pattern for managing multiple model
architectures and a factory function for creating models based on configuration.

Usage:
    from model_foundry.architectures import create_model_from_config
    model = create_model_from_config(config)
"""

from typing import Dict, Type, Callable
from .base import BaseLanguageModel, ModelOutput

# Global registry mapping architecture names to model classes
MODEL_REGISTRY: Dict[str, Type[BaseLanguageModel]] = {}


def register_architecture(name: str) -> Callable:
    """
    Decorator to register model architectures.

    This decorator registers a model class in the global MODEL_REGISTRY,
    making it available for creation via the factory function.

    Args:
        name: Architecture identifier (e.g., "gpt2", "bert", "lstm")

    Returns:
        Decorator function

    Example:
        @register_architecture("gpt2")
        class GPT2Model(BaseLanguageModel):
            ...

    Raises:
        ValueError: If architecture name is already registered
    """
    def decorator(cls: Type[BaseLanguageModel]) -> Type[BaseLanguageModel]:
        if name in MODEL_REGISTRY:
            raise ValueError(
                f"Architecture '{name}' is already registered. "
                f"Existing class: {MODEL_REGISTRY[name].__name__}"
            )

        # Verify the class implements BaseLanguageModel
        if not issubclass(cls, BaseLanguageModel):
            raise TypeError(
                f"Class {cls.__name__} must inherit from BaseLanguageModel"
            )

        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_registered_architectures() -> list:
    """
    Get list of all registered architecture names.

    Returns:
        List of registered architecture identifiers
    """
    return list(MODEL_REGISTRY.keys())


def create_model_from_config(config, **kwargs) -> BaseLanguageModel:
    """
    Factory function to create models based on configuration.

    This is the main entry point for model creation. It looks up the
    appropriate model class from the registry and instantiates it with
    the provided configuration.

    Args:
        config: ExperimentConfig object containing model configuration
        **kwargs: Additional arguments passed to model constructor
                  (e.g., attn_implementation for transformers)

    Returns:
        Instance of the appropriate model architecture

    Raises:
        ValueError: If architecture is not specified in config
        ValueError: If architecture is not registered
        AttributeError: If config doesn't have required model field

    Example:
        config = load_config("configs/experiment_gpt2.yaml")
        model = create_model_from_config(config)
        print(f"Created {model.model_type} model")
    """
    # Validate config has model configuration
    if not hasattr(config, 'model'):
        raise AttributeError(
            "Configuration must have 'model' field. "
            "Check your ExperimentConfig structure."
        )

    # Get architecture from config
    if not hasattr(config.model, 'architecture'):
        raise ValueError(
            "Configuration must specify 'architecture' field in model config. "
            f"Available architectures: {get_registered_architectures()}\n"
            "Example:\n"
            "  model:\n"
            "    architecture: 'gpt2'\n"
            "    transformer:\n"
            "      layers: 12\n"
            "      ..."
        )

    architecture = config.model.architecture

    # Look up model class in registry
    if architecture not in MODEL_REGISTRY:
        available = get_registered_architectures()
        raise ValueError(
            f"Unknown architecture: '{architecture}'. "
            f"Available architectures: {available}\n"
            f"Make sure the architecture module is imported."
        )

    # Get model class and create instance
    model_class = MODEL_REGISTRY[architecture]

    print(f"--- Building {architecture.upper()} Model from Config ---")

    # Instantiate model with config
    model = model_class.from_config(config, **kwargs)

    # Print model statistics
    total_params = model.get_parameter_count()
    print(f"  - Successfully created {architecture} model")
    print(f"  - Total Parameters: {total_params:,}")

    memory_footprint = model.get_memory_footprint()
    print(f"  - Memory footprint: {memory_footprint['total'] / (1024**2):.2f} MB")

    return model


# Import architecture implementations to trigger registration
# These imports will populate the MODEL_REGISTRY
def _register_default_architectures():
    """
    Import and register default architectures.

    This function is called automatically when the module is imported.
    It ensures that all built-in architectures are registered and available.
    """
    try:
        from . import gpt  # noqa: F401
        print(f"  - Registered GPT-2 architecture")
    except ImportError as e:
        print(f"  - Warning: Could not import GPT-2 architecture: {e}")

    # Future architectures will be imported here
    # try:
    #     from . import bert  # noqa: F401
    #     print(f"  - Registered BERT architecture")
    # except ImportError:
    #     pass

    # try:
    #     from . import rnn  # noqa: F401
    #     print(f"  - Registered RNN/LSTM/GRU architectures")
    # except ImportError:
    #     pass


# Import architecture implementations for export
from .gpt import GPT2Model  # noqa: E402

# Export public API
__all__ = [
    'BaseLanguageModel',
    'ModelOutput',
    'register_architecture',
    'get_registered_architectures',
    'create_model_from_config',
    'MODEL_REGISTRY',
    'GPT2Model',
]
