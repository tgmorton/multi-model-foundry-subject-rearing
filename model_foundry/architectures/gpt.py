"""
GPT-2 causal language model implementation.

This module wraps HuggingFace's GPT-2 implementation to conform to
the BaseLanguageModel interface, enabling it to work within the
multi-architecture Model Foundry framework.
"""

from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from .base import BaseLanguageModel, ModelOutput
from . import register_architecture


@register_architecture("gpt2")
class GPT2Model(BaseLanguageModel):
    """
    GPT-2 causal language model.

    This class wraps HuggingFace's AutoModelForCausalLM to provide a GPT-2
    model that conforms to the BaseLanguageModel interface.

    The model is configured using the transformer section of the model config
    and supports all standard GPT-2 features including:
    - Causal (autoregressive) language modeling
    - Flash Attention (when available)
    - Gradient checkpointing
    - Text generation

    Attributes:
        hf_model: The underlying HuggingFace GPT-2 model
        config: The HuggingFace model configuration
    """

    def __init__(self, hf_model: AutoModelForCausalLM, hf_config: AutoConfig):
        """
        Initialize GPT-2 model.

        Args:
            hf_model: HuggingFace GPT-2 model instance
            hf_config: HuggingFace model configuration
        """
        super().__init__()
        self.hf_model = hf_model
        self.config = hf_config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> ModelOutput:
        """
        Forward pass through GPT-2 model.

        Args:
            input_ids: Token IDs (batch_size, sequence_length)
            attention_mask: Attention mask (batch_size, sequence_length)
            labels: Target labels for causal LM loss (batch_size, sequence_length)
            **kwargs: Additional arguments for HuggingFace model

        Returns:
            ModelOutput with loss (if labels provided), logits, hidden_states, attentions

        Note:
            For causal language modeling, the labels are typically the same as
            input_ids. The model internally shifts labels by 1 position to
            predict the next token.
        """
        # Delegate to HuggingFace model
        outputs = self.hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=kwargs.get('output_hidden_states', False),
            output_attentions=kwargs.get('output_attentions', False),
            **{k: v for k, v in kwargs.items()
               if k not in ['output_hidden_states', 'output_attentions']}
        )

        # Convert HuggingFace output to our ModelOutput format
        return ModelOutput(
            loss=outputs.loss if hasattr(outputs, 'loss') else None,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )

    def get_input_embeddings(self) -> nn.Module:
        """
        Get the input embedding layer.

        Returns:
            The embedding layer from the GPT-2 model
        """
        return self.hf_model.get_input_embeddings()

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        """
        Resize token embeddings to match new vocabulary size.

        Args:
            new_num_tokens: New vocabulary size
        """
        self.hf_model.resize_token_embeddings(new_num_tokens)
        # Update config to reflect new vocab size
        self.config.vocab_size = new_num_tokens

    @property
    def model_type(self) -> str:
        """Return model type identifier."""
        return "gpt2"

    @property
    def supports_generation(self) -> bool:
        """GPT-2 supports autoregressive generation."""
        return True

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save model using HuggingFace's save method.

        Args:
            save_directory: Directory to save model files
        """
        self.hf_model.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, model_directory: str, **kwargs):
        """
        Load GPT-2 model from saved directory.

        Args:
            model_directory: Directory containing saved model
            **kwargs: Additional arguments for HuggingFace model loading

        Returns:
            Loaded GPT2Model instance
        """
        hf_model = AutoModelForCausalLM.from_pretrained(model_directory, **kwargs)
        hf_config = hf_model.config
        return cls(hf_model=hf_model, hf_config=hf_config)

    @classmethod
    def from_config(cls, config, **kwargs):
        """
        Create GPT-2 model from ExperimentConfig.

        This is the main factory method used by the model creation pipeline.
        It extracts relevant parameters from the config and creates a new
        GPT-2 model with random initialization.

        Args:
            config: ExperimentConfig object containing model, data, and tokenizer configs
            **kwargs: Additional arguments (e.g., attn_implementation for Flash Attention)

        Returns:
            New GPT2Model instance with random weights

        Raises:
            ValueError: If config doesn't have required transformer section
        """
        model_params = config.model
        tokenizer_params = config.tokenizer
        data_params = config.data

        # Validate transformer config exists
        if not hasattr(model_params, 'transformer') or model_params.transformer is None:
            raise ValueError(
                "GPT-2 architecture requires 'transformer' configuration. "
                "Example:\n"
                "  model:\n"
                "    architecture: 'gpt2'\n"
                "    transformer:\n"
                "      layers: 12\n"
                "      embedding_size: 768\n"
                "      ..."
            )

        transformer_config = model_params.transformer

        # Create HuggingFace GPT-2 config
        hf_config = AutoConfig.from_pretrained("gpt2")

        # Override with our parameters
        hf_config.n_layer = transformer_config.layers
        hf_config.n_embd = transformer_config.embedding_size
        hf_config.n_head = transformer_config.attention_heads
        hf_config.n_inner = transformer_config.intermediate_hidden_size
        hf_config.n_positions = data_params.max_sequence_length
        hf_config.activation_function = transformer_config.activation_function
        hf_config.resid_pdrop = transformer_config.dropout
        hf_config.embd_pdrop = transformer_config.dropout
        hf_config.attn_pdrop = transformer_config.attention_dropout
        hf_config.vocab_size = tokenizer_params.vocab_size
        hf_config.use_cache = False  # Disable KV cache for training

        print(f"  - Model vocabulary size set to: {hf_config.vocab_size}")

        # Set attention implementation if provided
        if 'attn_implementation' in kwargs:
            hf_config.attn_implementation = kwargs['attn_implementation']
            print(f"  - Set attention implementation to: {kwargs['attn_implementation']}")

        # Create model from config
        hf_model = AutoModelForCausalLM.from_config(hf_config, **kwargs)

        # Wrap in our GPT2Model class
        return cls(hf_model=hf_model, hf_config=hf_config)

    def generate(self, *args, **kwargs):
        """
        Generate text using the GPT-2 model.

        This delegates to HuggingFace's generate method, providing access
        to all generation strategies (greedy, beam search, sampling, etc.)

        Args:
            *args: Positional arguments for HuggingFace generate
            **kwargs: Keyword arguments for HuggingFace generate

        Returns:
            Generated token IDs

        Example:
            input_ids = tokenizer.encode("Once upon a time", return_tensors="pt")
            output_ids = model.generate(input_ids, max_length=100)
            generated_text = tokenizer.decode(output_ids[0])
        """
        return self.hf_model.generate(*args, **kwargs)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to reduce memory usage during training."""
        self.hf_model.gradient_checkpointing_enable()

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        if hasattr(self.hf_model, 'gradient_checkpointing_disable'):
            self.hf_model.gradient_checkpointing_disable()
