"""
LSTM/RNN architecture implementations for the Model Foundry framework.

This module provides RNN-based architectures (LSTM, GRU, vanilla RNN) for language modeling.
These models can operate in either:
- Unidirectional mode (causal LM): For autoregressive generation
- Bidirectional mode (masked LM): For masked language modeling

Unlike transformers, RNNs process sequences sequentially and have different memory
characteristics, making them suitable for certain linguistic phenomena studies.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLanguageModel, ModelOutput
from . import register_architecture


class RNNLanguageModel(BaseLanguageModel):
    """
    Base RNN-based language model supporting LSTM, GRU, and vanilla RNN.

    This implementation provides a flexible RNN architecture that can be configured
    for different cell types and directionality. It supports both causal (unidirectional)
    and masked (bidirectional) language modeling objectives.

    Architecture:
        - Token Embedding Layer
        - RNN Layers (LSTM/GRU/RNN, uni/bidirectional)
        - Dropout
        - Output Projection to Vocabulary

    Attributes:
        embedding: Token embedding layer
        rnn: RNN module (LSTM/GRU/RNN)
        dropout: Dropout layer
        output_projection: Linear layer mapping hidden states to vocabulary
        bidirectional: Whether the RNN is bidirectional
        rnn_type: Type of RNN cell ('lstm', 'gru', 'rnn')
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        rnn_type: str = "lstm",
        bidirectional: bool = False,
        dropout: float = 0.0,
        pad_token_id: int = 0,
        **kwargs
    ):
        """
        Initialize RNN language model.

        Args:
            vocab_size: Size of vocabulary
            embedding_size: Dimension of token embeddings
            hidden_size: Hidden state dimension for RNN
            num_layers: Number of stacked RNN layers
            rnn_type: Type of RNN cell ('lstm', 'gru', 'rnn')
            bidirectional: If True, use bidirectional RNN (for masked LM)
            dropout: Dropout probability between RNN layers
            pad_token_id: Token ID for padding (default: 0)
            **kwargs: Additional arguments (for compatibility)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.dropout_prob = dropout
        self.pad_token_id = pad_token_id

        # Validate RNN type
        if self.rnn_type not in ['lstm', 'gru', 'rnn']:
            raise ValueError(f"rnn_type must be 'lstm', 'gru', or 'rnn', got '{rnn_type}'")

        # Token embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_size,
            padding_idx=pad_token_id
        )

        # Select RNN class based on type
        if self.rnn_type == 'lstm':
            rnn_class = nn.LSTM
        elif self.rnn_type == 'gru':
            rnn_class = nn.GRU
        else:  # 'rnn'
            rnn_class = nn.RNN

        # RNN layers
        # Note: dropout is applied between layers (not on last layer output)
        self.rnn = rnn_class(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # Dropout layer (applied to RNN output)
        self.dropout = nn.Dropout(dropout)

        # Output projection
        # If bidirectional, hidden size is doubled
        projection_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.output_projection = nn.Linear(projection_input_size, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.pad_token_id is not None:
            # Zero out padding token embedding
            with torch.no_grad():
                self.embedding.weight[self.pad_token_id].fill_(0.0)

        # Initialize output projection
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> ModelOutput:
        """
        Forward pass through RNN model.

        Args:
            input_ids: Token IDs [batch_size, sequence_length]
            attention_mask: Attention mask [batch_size, sequence_length]
                           1 for real tokens, 0 for padding
            labels: Labels for language modeling [batch_size, sequence_length]
                   For causal LM: typically same as input_ids (shifted internally)
                   For masked LM: -100 for non-masked positions
            **kwargs: Additional arguments (ignored)

        Returns:
            ModelOutput with loss and logits
        """
        batch_size, seq_length = input_ids.shape

        # Get embeddings
        embeddings = self.embedding(input_ids)  # [batch, seq, emb]

        # For efficiency with variable-length sequences, use pack_padded_sequence
        if attention_mask is not None:
            # Compute actual lengths (number of non-padding tokens)
            lengths = attention_mask.sum(dim=1).cpu()  # [batch]

            # Check if any sequence has length > 0 and has padding
            has_padding = (lengths < seq_length).any()
            all_zero = (lengths == 0).all()

            # Only pack if there's actual padding and at least one non-zero sequence
            if has_padding and not all_zero:
                # Ensure no zero-length sequences (clamp to minimum of 1)
                # This handles edge cases where all tokens might be padding
                lengths = lengths.clamp(min=1)

                # Pack sequences
                embeddings = nn.utils.rnn.pack_padded_sequence(
                    embeddings,
                    lengths,
                    batch_first=True,
                    enforce_sorted=False
                )

                # RNN forward pass
                rnn_output, _ = self.rnn(embeddings)

                # Unpack sequences
                rnn_output, _ = nn.utils.rnn.pad_packed_sequence(
                    rnn_output,
                    batch_first=True,
                    total_length=seq_length
                )
            else:
                # No padding, process normally (or all padding)
                rnn_output, _ = self.rnn(embeddings)
        else:
            # No attention mask provided, process entire sequence
            rnn_output, _ = self.rnn(embeddings)

        # Apply dropout
        rnn_output = self.dropout(rnn_output)  # [batch, seq, hidden*directions]

        # Project to vocabulary
        logits = self.output_projection(rnn_output)  # [batch, seq, vocab]

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Flatten for cross-entropy
            # logits: [batch*seq, vocab]
            # labels: [batch*seq]
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100  # Ignore positions with label -100
            )

        return ModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=rnn_output,
            attentions=None  # RNNs don't have attention
        )

    def get_input_embeddings(self) -> nn.Module:
        """
        Get the input embedding layer.

        Returns:
            Token embedding module
        """
        return self.embedding

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        """
        Resize token embeddings to accommodate new vocabulary size.

        Args:
            new_num_tokens: New vocabulary size
        """
        old_num_tokens = self.vocab_size

        if new_num_tokens == old_num_tokens:
            return

        # Create new embedding layer
        new_embedding = nn.Embedding(
            new_num_tokens,
            self.embedding_size,
            padding_idx=self.pad_token_id
        )

        # Copy old embeddings
        num_to_copy = min(old_num_tokens, new_num_tokens)
        new_embedding.weight.data[:num_to_copy] = self.embedding.weight.data[:num_to_copy]

        # Initialize new embeddings if expanded
        if new_num_tokens > old_num_tokens:
            nn.init.normal_(
                new_embedding.weight.data[old_num_tokens:],
                mean=0.0,
                std=0.02
            )

        # Replace embedding layer
        self.embedding = new_embedding

        # Resize output projection
        new_output = nn.Linear(
            self.output_projection.in_features,
            new_num_tokens
        )

        # Copy old weights
        new_output.weight.data[:num_to_copy] = self.output_projection.weight.data[:num_to_copy]
        if self.output_projection.bias is not None:
            new_output.bias.data[:num_to_copy] = self.output_projection.bias.data[:num_to_copy]

        # Initialize new weights if expanded
        if new_num_tokens > old_num_tokens:
            nn.init.normal_(
                new_output.weight.data[old_num_tokens:],
                mean=0.0,
                std=0.02
            )
            if new_output.bias is not None:
                nn.init.zeros_(new_output.bias.data[old_num_tokens:])

        self.output_projection = new_output
        self.vocab_size = new_num_tokens

    @property
    def model_type(self) -> str:
        """Get model type identifier."""
        # Return specific type based on RNN cell
        return self.rnn_type

    @property
    def supports_generation(self) -> bool:
        """
        Whether model supports autoregressive generation.

        Returns:
            True for unidirectional RNNs (can generate)
            False for bidirectional RNNs (cannot generate)
        """
        return not self.bidirectional

    @classmethod
    def from_config(cls, config, **kwargs):
        """
        Create RNN model from experiment configuration.

        This method extracts RNN-specific parameters from the config and creates
        an RNN language model with the specified architecture.

        Args:
            config: ExperimentConfig with model.architecture in ['lstm', 'gru', 'rnn']
            **kwargs: Additional arguments for model initialization

        Returns:
            RNNLanguageModel instance

        Raises:
            ValueError: If config doesn't specify RNN architecture or rnn config
        """
        arch = config.model.architecture
        if arch not in ['lstm', 'gru', 'rnn']:
            raise ValueError(
                f"RNNLanguageModel.from_config() called with architecture '{arch}'. "
                "Expected 'lstm', 'gru', or 'rnn'."
            )

        if config.model.rnn is None:
            raise ValueError(
                f"{arch.upper()} architecture requires 'rnn' configuration. "
                "Please specify model.rnn in your config."
            )

        # Extract RNN configuration
        rnn_config = config.model.rnn

        # Get vocabulary size from tokenizer config
        vocab_size = config.tokenizer.vocab_size

        # Determine pad token ID
        pad_token_id = 0  # Default
        if hasattr(config.tokenizer, 'special_tokens') and config.tokenizer.special_tokens:
            # Try to extract pad token ID from special tokens
            # This would require tokenizer to be loaded, so we use default for now
            pass

        # Create model
        return cls(
            vocab_size=vocab_size,
            embedding_size=rnn_config.embedding_size,
            hidden_size=rnn_config.hidden_size,
            num_layers=rnn_config.num_layers,
            rnn_type=rnn_config.rnn_type,
            bidirectional=rnn_config.bidirectional,
            dropout=rnn_config.dropout,
            pad_token_id=pad_token_id,
            **kwargs
        )


# Register specific RNN types
@register_architecture("lstm")
class LSTMModel(RNNLanguageModel):
    """
    LSTM (Long Short-Term Memory) language model.

    LSTMs are a type of RNN with gating mechanisms that help with long-range
    dependencies. They maintain both a hidden state and a cell state.

    Can be configured as:
    - Unidirectional (causal LM): For autoregressive language modeling
    - Bidirectional (masked LM): For masked language modeling like BERT
    """

    def __init__(self, *args, **kwargs):
        """Initialize LSTM model, ensuring rnn_type is lstm."""
        if 'rnn_type' in kwargs and kwargs['rnn_type'] != 'lstm':
            kwargs['rnn_type'] = 'lstm'
        elif 'rnn_type' not in kwargs:
            kwargs['rnn_type'] = 'lstm'
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config, **kwargs):
        """Create LSTM model from config."""
        arch = config.model.architecture
        if arch != 'lstm':
            raise ValueError(
                f"LSTMModel.from_config() called with architecture '{arch}'. "
                "Expected 'lstm'."
            )

        if config.model.rnn is None:
            raise ValueError(
                "LSTM architecture requires 'rnn' configuration. "
                "Please specify model.rnn in your config."
            )

        # Extract RNN configuration
        rnn_config = config.model.rnn

        # Get vocabulary size from tokenizer config
        vocab_size = config.tokenizer.vocab_size

        # Determine pad token ID
        pad_token_id = 0  # Default

        # Create LSTM model
        return cls(
            vocab_size=vocab_size,
            embedding_size=rnn_config.embedding_size,
            hidden_size=rnn_config.hidden_size,
            num_layers=rnn_config.num_layers,
            rnn_type='lstm',
            bidirectional=rnn_config.bidirectional,
            dropout=rnn_config.dropout,
            pad_token_id=pad_token_id,
            **kwargs
        )


@register_architecture("gru")
class GRUModel(RNNLanguageModel):
    """
    GRU (Gated Recurrent Unit) language model.

    GRUs are a simplified variant of LSTMs with fewer parameters. They have
    update and reset gates but no separate cell state.

    Can be configured as:
    - Unidirectional (causal LM): For autoregressive language modeling
    - Bidirectional (masked LM): For masked language modeling like BERT
    """

    def __init__(self, *args, **kwargs):
        """Initialize GRU model, ensuring rnn_type is gru."""
        if 'rnn_type' in kwargs and kwargs['rnn_type'] != 'gru':
            kwargs['rnn_type'] = 'gru'
        elif 'rnn_type' not in kwargs:
            kwargs['rnn_type'] = 'gru'
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config, **kwargs):
        """Create GRU model from config."""
        arch = config.model.architecture
        if arch != 'gru':
            raise ValueError(
                f"GRUModel.from_config() called with architecture '{arch}'. "
                "Expected 'gru'."
            )

        if config.model.rnn is None:
            raise ValueError(
                "GRU architecture requires 'rnn' configuration. "
                "Please specify model.rnn in your config."
            )

        # Extract RNN configuration
        rnn_config = config.model.rnn

        # Get vocabulary size from tokenizer config
        vocab_size = config.tokenizer.vocab_size

        # Determine pad token ID
        pad_token_id = 0  # Default

        # Create GRU model
        return cls(
            vocab_size=vocab_size,
            embedding_size=rnn_config.embedding_size,
            hidden_size=rnn_config.hidden_size,
            num_layers=rnn_config.num_layers,
            rnn_type='gru',
            bidirectional=rnn_config.bidirectional,
            dropout=rnn_config.dropout,
            pad_token_id=pad_token_id,
            **kwargs
        )


@register_architecture("rnn")
class VanillaRNNModel(RNNLanguageModel):
    """
    Vanilla RNN (Elman RNN) language model.

    The simplest form of RNN with just a single tanh activation. Prone to
    vanishing/exploding gradients but useful for studying basic sequence modeling.

    Can be configured as:
    - Unidirectional (causal LM): For autoregressive language modeling
    - Bidirectional (masked LM): For masked language modeling like BERT
    """

    def __init__(self, *args, **kwargs):
        """Initialize vanilla RNN model, ensuring rnn_type is rnn."""
        if 'rnn_type' in kwargs and kwargs['rnn_type'] != 'rnn':
            kwargs['rnn_type'] = 'rnn'
        elif 'rnn_type' not in kwargs:
            kwargs['rnn_type'] = 'rnn'
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config, **kwargs):
        """Create vanilla RNN model from config."""
        arch = config.model.architecture
        if arch != 'rnn':
            raise ValueError(
                f"VanillaRNNModel.from_config() called with architecture '{arch}'. "
                "Expected 'rnn'."
            )

        if config.model.rnn is None:
            raise ValueError(
                "RNN architecture requires 'rnn' configuration. "
                "Please specify model.rnn in your config."
            )

        # Extract RNN configuration
        rnn_config = config.model.rnn

        # Get vocabulary size from tokenizer config
        vocab_size = config.tokenizer.vocab_size

        # Determine pad token ID
        pad_token_id = 0  # Default

        # Create vanilla RNN model
        return cls(
            vocab_size=vocab_size,
            embedding_size=rnn_config.embedding_size,
            hidden_size=rnn_config.hidden_size,
            num_layers=rnn_config.num_layers,
            rnn_type='rnn',
            bidirectional=rnn_config.bidirectional,
            dropout=rnn_config.dropout,
            pad_token_id=pad_token_id,
            **kwargs
        )
