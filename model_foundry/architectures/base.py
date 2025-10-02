"""
Base classes for multi-architecture language model support.

This module defines the abstract interfaces that all language model
architectures must implement to work within the Model Foundry framework.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch
import torch.nn as nn


class ModelOutput:
    """
    Standardized output format for all model architectures.

    This class provides a unified interface for model outputs, compatible
    with HuggingFace's ModelOutput but simplified for our use case.

    Attributes:
        loss: Training loss (if labels provided)
        logits: Model output logits (batch_size, sequence_length, vocab_size)
        hidden_states: Hidden states from model layers (optional)
        attentions: Attention weights (optional, not available for RNNs)
    """

    def __init__(
        self,
        loss: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attentions: Optional[torch.Tensor] = None,
    ):
        self.loss = loss
        self.logits = logits
        self.hidden_states = hidden_states
        self.attentions = attentions

    def __getitem__(self, key: str):
        """Allow dictionary-style access for HuggingFace compatibility."""
        return getattr(self, key, None)

    def __contains__(self, key: str) -> bool:
        """Check if attribute exists."""
        return hasattr(self, key)


class BaseLanguageModel(nn.Module, ABC):
    """
    Abstract base class for all language model architectures.

    All model implementations (GPT-2, BERT, LSTM, etc.) must inherit from
    this class and implement its abstract methods. This ensures a consistent
    interface across all architectures.

    The base class is compatible with PyTorch's nn.Module and can be used
    with standard PyTorch training loops, optimizers, and utilities.
    """

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> ModelOutput:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs (batch_size, sequence_length)
            attention_mask: Attention mask (batch_size, sequence_length)
            labels: Target labels for loss computation (batch_size, sequence_length)
            **kwargs: Additional architecture-specific arguments

        Returns:
            ModelOutput containing:
                - loss (if labels provided)
                - logits (batch_size, sequence_length, vocab_size)
                - hidden_states (optional)
                - attentions (optional)

        Note:
            The loss computation should be handled internally by each
            architecture based on its training objective (causal LM, masked LM, etc.)
        """
        pass

    @abstractmethod
    def get_input_embeddings(self) -> nn.Module:
        """
        Return the input embedding layer.

        This is used for:
        - Vocabulary resizing
        - Weight tying with output layer
        - Initialization

        Returns:
            The embedding layer (nn.Embedding or equivalent)
        """
        pass

    @abstractmethod
    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        """
        Resize the model's token embeddings to match a new vocabulary size.

        This is necessary when:
        - Loading a model with a different tokenizer
        - Adding special tokens to the vocabulary

        Args:
            new_num_tokens: The new vocabulary size

        Note:
            Implementations should preserve existing embeddings and initialize
            new embeddings appropriately.
        """
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        """
        Return the model architecture type identifier.

        Returns:
            Architecture identifier (e.g., "gpt2", "bert", "lstm", "gru", "rnn")

        This is used for:
        - Model registration and factory lookup
        - Logging and checkpointing
        - Architecture-specific evaluation
        """
        pass

    @property
    @abstractmethod
    def supports_generation(self) -> bool:
        """
        Whether the model supports autoregressive text generation.

        Returns:
            True if model can generate text autoregressively, False otherwise

        Models that support generation:
        - Causal language models (GPT-2, unidirectional LSTM)

        Models that don't support generation:
        - Masked language models (BERT)
        - Bidirectional models (bidirectional LSTM)

        This is used to determine:
        - Which evaluation methods are applicable
        - Whether generation utilities can be used
        """
        pass

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save model weights and configuration to directory.

        Default implementation saves PyTorch state dict. Architectures
        using HuggingFace models should override to use HF's save method.

        Args:
            save_directory: Directory to save model files
        """
        import os
        os.makedirs(save_directory, exist_ok=True)

        # Save state dict
        state_dict_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), state_dict_path)

        # Save model type info
        config_path = os.path.join(save_directory, "model_type.txt")
        with open(config_path, 'w') as f:
            f.write(self.model_type)

    @classmethod
    def from_pretrained(cls, model_directory: str, **kwargs):
        """
        Load model from saved directory.

        Default implementation loads PyTorch state dict. Architectures
        using HuggingFace models should override to use HF's load method.

        Args:
            model_directory: Directory containing saved model files
            **kwargs: Additional arguments for model initialization

        Returns:
            Loaded model instance
        """
        import os

        # Load state dict
        state_dict_path = os.path.join(model_directory, "pytorch_model.bin")
        state_dict = torch.load(state_dict_path, map_location="cpu")

        # Create model instance and load weights
        # Subclasses should override this method for proper initialization
        raise NotImplementedError(
            f"{cls.__name__} must implement from_pretrained method"
        )

    def get_parameter_count(self) -> int:
        """
        Get total number of trainable parameters.

        Returns:
            Total number of parameters in the model
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_memory_footprint(self) -> Dict[str, int]:
        """
        Estimate model memory footprint.

        Returns:
            Dictionary with memory statistics in bytes
        """
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())

        return {
            "parameters": param_size,
            "buffers": buffer_size,
            "total": param_size + buffer_size,
        }
