"""
BERT architecture implementation for the Model Foundry framework.

This module provides a BERT (Bidirectional Encoder Representations from Transformers)
implementation using HuggingFace's transformers library, wrapped to conform to the
BaseLanguageModel interface.

BERT is trained using masked language modeling (MLM) where tokens are randomly masked
and the model learns to predict them based on bidirectional context.
"""

from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForMaskedLM

from .base import BaseLanguageModel, ModelOutput
from . import register_architecture


@register_architecture("bert")
class BERTModel(BaseLanguageModel):
    """
    BERT model wrapper for masked language modeling.

    This implementation wraps HuggingFace's BERT model and provides the standard
    BaseLanguageModel interface for use in the Model Foundry training pipeline.

    Key differences from GPT-2:
    - Uses bidirectional attention (can see full context)
    - Trained with masked language modeling objective
    - Includes segment embeddings for sentence pairs
    - Uses [CLS] token for sequence representation

    Attributes:
        hf_model: The underlying HuggingFace BERT model
        config: HuggingFace model configuration
    """

    def __init__(self, hf_model: AutoModelForMaskedLM, hf_config: AutoConfig):
        """
        Initialize BERT model wrapper.

        Args:
            hf_model: HuggingFace BERT model for masked LM
            hf_config: HuggingFace configuration object
        """
        super().__init__()
        self.hf_model = hf_model
        self.config = hf_config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> ModelOutput:
        """
        Forward pass through BERT model.

        Args:
            input_ids: Token IDs [batch_size, sequence_length]
            attention_mask: Attention mask [batch_size, sequence_length]
            labels: Labels for masked language modeling [batch_size, sequence_length]
                   -100 for tokens that should not be predicted
            token_type_ids: Segment IDs for sentence pairs [batch_size, sequence_length]
            **kwargs: Additional arguments passed to HuggingFace model

        Returns:
            ModelOutput with loss, logits, hidden_states, and attentions
        """
        outputs = self.hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            token_type_ids=token_type_ids,
            **kwargs
        )

        return ModelOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    def get_input_embeddings(self) -> nn.Module:
        """
        Get the input embedding layer.

        Returns:
            Input embedding module
        """
        return self.hf_model.get_input_embeddings()

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        """
        Resize token embeddings to accommodate new vocabulary size.

        Args:
            new_num_tokens: New vocabulary size
        """
        self.hf_model.resize_token_embeddings(new_num_tokens)

    @property
    def model_type(self) -> str:
        """Get model type identifier."""
        return "bert"

    @property
    def supports_generation(self) -> bool:
        """
        BERT does not support autoregressive generation.

        Returns:
            False - BERT is bidirectional and cannot generate autoregressively
        """
        return False

    @classmethod
    def from_config(cls, config, **kwargs):
        """
        Create a BERT model from experiment configuration.

        This method extracts BERT-specific parameters from the config and creates
        a HuggingFace BERT model with the specified architecture.

        Args:
            config: ExperimentConfig with model.architecture == "bert"
            **kwargs: Additional arguments for model initialization

        Returns:
            BERTModel instance

        Raises:
            ValueError: If config doesn't specify BERT architecture or transformer config
        """
        if config.model.architecture != "bert":
            raise ValueError(
                f"BERTModel.from_config() called with architecture '{config.model.architecture}'. "
                "Expected 'bert'."
            )

        if config.model.transformer is None:
            raise ValueError(
                "BERT architecture requires 'transformer' configuration. "
                "Please specify model.transformer in your config."
            )

        # Extract transformer configuration
        transformer_config = config.model.transformer

        # Create HuggingFace BERT config
        hf_config = AutoConfig.from_pretrained("bert-base-uncased")

        # Map our config to HuggingFace BERT config
        hf_config.num_hidden_layers = transformer_config.layers
        hf_config.hidden_size = transformer_config.hidden_size
        hf_config.intermediate_size = transformer_config.intermediate_hidden_size
        hf_config.num_attention_heads = transformer_config.attention_heads
        hf_config.hidden_dropout_prob = transformer_config.dropout
        hf_config.attention_probs_dropout_prob = transformer_config.attention_dropout
        hf_config.hidden_act = transformer_config.activation_function

        # BERT-specific configurations
        if config.model.bert is not None:
            hf_config.type_vocab_size = config.model.bert.type_vocab_size

        # Set vocabulary size from tokenizer config if available
        if hasattr(config, 'tokenizer') and hasattr(config.tokenizer, 'vocab_size'):
            hf_config.vocab_size = config.tokenizer.vocab_size

        # Set max position embeddings from data config if available
        if hasattr(config, 'data') and hasattr(config.data, 'max_sequence_length'):
            hf_config.max_position_embeddings = config.data.max_sequence_length

        # Create model from config
        hf_model = AutoModelForMaskedLM.from_config(hf_config, **kwargs)

        return cls(hf_model=hf_model, hf_config=hf_config)
