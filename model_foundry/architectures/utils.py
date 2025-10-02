"""
Shared utilities for model architectures.

This module provides common functionality used across different
model architectures, such as parameter initialization, model
inspection, and compatibility utilities.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


def get_model_statistics(model: nn.Module) -> Dict[str, Any]:
    """
    Get comprehensive statistics about a model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary containing:
            - total_params: Total parameter count
            - trainable_params: Trainable parameter count
            - non_trainable_params: Non-trainable parameter count
            - param_size_mb: Total parameter size in MB
            - buffer_size_mb: Total buffer size in MB
            - total_size_mb: Total model size in MB
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_size_mb": param_size / (1024 ** 2),
        "buffer_size_mb": buffer_size / (1024 ** 2),
        "total_size_mb": (param_size + buffer_size) / (1024 ** 2),
    }


def initialize_embeddings(
    embedding_layer: nn.Embedding,
    mean: float = 0.0,
    std: float = 0.02
) -> None:
    """
    Initialize embedding layer with normal distribution.

    This follows the standard practice for transformer models.

    Args:
        embedding_layer: The embedding layer to initialize
        mean: Mean of the normal distribution
        std: Standard deviation of the normal distribution
    """
    nn.init.normal_(embedding_layer.weight, mean=mean, std=std)

    # Zero out padding token embedding if it exists
    if embedding_layer.padding_idx is not None:
        with torch.no_grad():
            embedding_layer.weight[embedding_layer.padding_idx].fill_(0)


def initialize_linear_layer(
    linear_layer: nn.Linear,
    mean: float = 0.0,
    std: float = 0.02
) -> None:
    """
    Initialize linear layer with normal distribution.

    Args:
        linear_layer: The linear layer to initialize
        mean: Mean of the normal distribution
        std: Standard deviation of the normal distribution
    """
    nn.init.normal_(linear_layer.weight, mean=mean, std=std)
    if linear_layer.bias is not None:
        nn.init.zeros_(linear_layer.bias)


def resize_embeddings_with_preservation(
    embedding_layer: nn.Embedding,
    new_vocab_size: int,
    mean: float = 0.0,
    std: float = 0.02
) -> nn.Embedding:
    """
    Resize embedding layer while preserving existing embeddings.

    When expanding vocabulary, new embeddings are initialized randomly.
    When shrinking, embeddings are truncated.

    Args:
        embedding_layer: Existing embedding layer
        new_vocab_size: New vocabulary size
        mean: Mean for initializing new embeddings
        std: Standard deviation for initializing new embeddings

    Returns:
        New embedding layer with updated size
    """
    old_vocab_size = embedding_layer.num_embeddings
    embedding_dim = embedding_layer.embedding_dim
    padding_idx = embedding_layer.padding_idx

    # Create new embedding layer
    new_embedding = nn.Embedding(
        new_vocab_size,
        embedding_dim,
        padding_idx=padding_idx
    )

    # Initialize all embeddings
    nn.init.normal_(new_embedding.weight, mean=mean, std=std)

    # Copy existing embeddings
    num_to_copy = min(old_vocab_size, new_vocab_size)
    with torch.no_grad():
        new_embedding.weight[:num_to_copy] = embedding_layer.weight[:num_to_copy]

        # Ensure padding token is zero
        if padding_idx is not None and padding_idx < new_vocab_size:
            new_embedding.weight[padding_idx].fill_(0)

    return new_embedding


def check_model_device_consistency(model: nn.Module) -> bool:
    """
    Check if all model parameters are on the same device.

    Args:
        model: PyTorch model to check

    Returns:
        True if all parameters are on the same device, False otherwise
    """
    devices = {p.device for p in model.parameters()}
    return len(devices) <= 1


def get_model_device(model: nn.Module) -> Optional[torch.device]:
    """
    Get the device that the model is on.

    Args:
        model: PyTorch model

    Returns:
        The device, or None if model has no parameters
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


def count_parameters_by_type(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters grouped by layer type.

    Args:
        model: PyTorch model

    Returns:
        Dictionary mapping layer types to parameter counts
    """
    param_counts = {}

    for name, module in model.named_modules():
        module_type = module.__class__.__name__

        if module_type not in param_counts:
            param_counts[module_type] = 0

        # Count direct parameters of this module (not submodules)
        for param in module.parameters(recurse=False):
            param_counts[module_type] += param.numel()

    return param_counts


def print_model_summary(model: nn.Module, model_name: str = "Model") -> None:
    """
    Print a comprehensive summary of the model.

    Args:
        model: PyTorch model
        model_name: Name to display in summary
    """
    print(f"\n{'=' * 70}")
    print(f"{model_name} Summary")
    print(f"{'=' * 70}")

    # Get statistics
    stats = get_model_statistics(model)

    print(f"Total parameters:      {stats['total_params']:,}")
    print(f"Trainable parameters:  {stats['trainable_params']:,}")
    print(f"Non-trainable params:  {stats['non_trainable_params']:,}")
    print(f"Parameter size:        {stats['param_size_mb']:.2f} MB")
    print(f"Buffer size:           {stats['buffer_size_mb']:.2f} MB")
    print(f"Total size:            {stats['total_size_mb']:.2f} MB")

    # Device info
    device = get_model_device(model)
    if device:
        print(f"Device:                {device}")

    # Layer type breakdown
    print(f"\nParameters by layer type:")
    param_by_type = count_parameters_by_type(model)
    for layer_type, count in sorted(param_by_type.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            percentage = (count / stats['total_params']) * 100
            print(f"  {layer_type:30s} {count:12,} ({percentage:5.1f}%)")

    print(f"{'=' * 70}\n")
