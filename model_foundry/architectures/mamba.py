"""
Mamba (Selective State Space Model) architecture implementation.

This module provides a Mamba implementation that:
- Uses official mamba-ssm package if available (Linux + CUDA)
- Falls back to pure PyTorch implementation otherwise (CPU, macOS)

Mamba is a state space model with selective mechanisms and linear-time complexity,
making it efficient for long sequence modeling.

References:
    Gu & Dao (2023): "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
"""

import warnings
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .base import BaseLanguageModel, ModelOutput
from . import register_architecture


# Try to import mamba-ssm (available on Linux + CUDA)
try:
    from mamba_ssm import Mamba as MambaSSM
    MAMBA_SSM_AVAILABLE = True
except ImportError:
    MAMBA_SSM_AVAILABLE = False
    MambaSSM = None


class MambaBlock(nn.Module):
    """
    Pure PyTorch implementation of Mamba block.

    This is a simplified version that works on CPU/GPU without custom CUDA kernels.
    Falls back to this when mamba-ssm is not available.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand

        # Linear projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Convolution (for local context)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # SSM parameters (simplified)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # SSM initialization
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x):
        """
        Forward pass through Mamba block.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape

        # Input projection and split
        x_and_res = self.in_proj(x)  # [batch, seq_len, d_inner * 2]
        x_ssm, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)

        # Convolution (process along sequence)
        x_conv = x_ssm.transpose(1, 2)  # [batch, d_inner, seq_len]
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Trim padding
        x_conv = x_conv.transpose(1, 2)  # [batch, seq_len, d_inner]

        x_ssm = F.silu(x_conv)

        # SSM parameters
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        D = self.D.float()

        # Simplified SSM computation (not the full selective mechanism)
        # This is a linearized approximation for CPU compatibility
        deltaBC = self.x_proj(x_ssm)  # [batch, seq_len, d_state * 2]
        delta, B_C = deltaBC.split([self.d_state, self.d_state], dim=-1)

        # Discretization
        dt = F.softplus(self.dt_proj(x_ssm))  # [batch, seq_len, d_inner]

        # Simplified state space evolution (sequential for correctness)
        # Note: This is the CPU-friendly version
        # The real mamba-ssm uses parallel scan with CUDA kernels
        y = self._selective_scan(x_ssm, dt, A, B_C, D)

        # Gating
        y = y * F.silu(res)

        # Output projection
        output = self.out_proj(y)

        return output

    def _selective_scan(self, x, dt, A, B, D):
        """
        Simplified selective scan.

        This is a basic sequential scan for CPU compatibility.
        The real mamba-ssm uses optimized parallel scan on GPU.
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        # Initialize state
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)

        outputs = []

        for t in range(seq_len):
            # Current input
            x_t = x[:, t, :]  # [batch, d_inner]
            dt_t = dt[:, t, :]  # [batch, d_inner]
            B_t = B[:, t, :].unsqueeze(1)  # [batch, 1, d_state]

            # Discretize: A_discrete = exp(dt * A)
            A_discrete = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))  # [batch, d_inner, d_state]

            # Update state: h = A_discrete * h + dt * B * x
            h = A_discrete * h + (dt_t.unsqueeze(-1) * B_t * x_t.unsqueeze(-1))

            # Output: y = sum(h) + D * x
            y_t = h.sum(dim=-1) + D.unsqueeze(0) * x_t
            outputs.append(y_t)

        # Stack outputs
        y = torch.stack(outputs, dim=1)  # [batch, seq_len, d_inner]

        return y


class MambaLayer(nn.Module):
    """
    Full Mamba layer with normalization and residual connection.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_fast: bool = False,
    ):
        super().__init__()

        self.use_fast = use_fast and MAMBA_SSM_AVAILABLE

        if self.use_fast:
            # Use optimized mamba-ssm implementation
            self.mixer = MambaSSM(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # Use pure PyTorch implementation
            if use_fast and not MAMBA_SSM_AVAILABLE:
                warnings.warn(
                    "mamba-ssm not available. Using pure PyTorch implementation. "
                    "Install mamba-ssm on Linux with CUDA for optimal performance."
                )
            self.mixer = MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        return self.mixer(self.norm(x)) + x


@register_architecture("mamba")
class MambaModel(BaseLanguageModel):
    """
    Mamba state space model for language modeling.

    Mamba uses selective state spaces for efficient sequence processing with
    linear time complexity. This implementation:
    - Uses mamba-ssm package if available (Linux + CUDA, optimal)
    - Falls back to pure PyTorch if not (CPU/macOS, functional but slower)

    Supports causal language modeling (autoregressive generation).

    Architecture:
        - Token embeddings
        - N Mamba layers (selective SSM with convolution)
        - Layer normalization
        - Output projection to vocabulary

    Key features:
        - O(n) time complexity (vs O(n²) for transformers)
        - Efficient for long sequences
        - Competitive performance with transformers
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        pad_token_id: int = 0,
        use_fast: bool = True,
        **kwargs
    ):
        """
        Initialize Mamba model.

        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension (embedding size)
            n_layers: Number of Mamba layers
            d_state: SSM state expansion factor
            d_conv: Convolution kernel size
            expand: Block expansion factor
            dropout: Dropout probability
            pad_token_id: Padding token ID
            use_fast: Try to use mamba-ssm if available (GPU)
            **kwargs: Additional arguments
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dropout_prob = dropout
        self.pad_token_id = pad_token_id
        self._use_fast = use_fast and MAMBA_SSM_AVAILABLE

        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)

        # Mamba layers
        self.layers = nn.ModuleList([
            MambaLayer(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                use_fast=use_fast,
            )
            for _ in range(n_layers)
        ])

        # Final norm
        self.norm_f = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Language model head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (common practice)
        self.lm_head.weight = self.embedding.weight

        # Initialize weights
        self._init_weights()

        # Log which implementation is being used
        impl_type = "mamba-ssm (optimized)" if self._use_fast else "PyTorch (fallback)"
        print(f"  - Mamba implementation: {impl_type}")

    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.pad_token_id is not None:
            with torch.no_grad():
                self.embedding.weight[self.pad_token_id].fill_(0.0)

        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> ModelOutput:
        """
        Forward pass through Mamba model.

        Args:
            input_ids: Token IDs [batch_size, sequence_length]
            attention_mask: Attention mask [batch_size, sequence_length] (optional)
            labels: Labels for language modeling [batch_size, sequence_length]
            **kwargs: Additional arguments (ignored)

        Returns:
            ModelOutput with loss and logits
        """
        batch_size, seq_length = input_ids.shape

        # Get embeddings
        hidden_states = self.embedding(input_ids)  # [batch, seq, d_model]
        hidden_states = self.dropout(hidden_states)

        # Pass through Mamba layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # Final normalization
        hidden_states = self.norm_f(hidden_states)

        # Language model head
        logits = self.lm_head(hidden_states)  # [batch, seq, vocab_size]

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )

        return ModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=None  # Mamba doesn't have attention
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
            self.d_model,
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

        # Resize LM head (tied weights)
        self.lm_head = nn.Linear(self.d_model, new_num_tokens, bias=False)
        self.lm_head.weight = self.embedding.weight

        self.vocab_size = new_num_tokens

    @property
    def model_type(self) -> str:
        """Get model type identifier."""
        return "mamba"

    @property
    def supports_generation(self) -> bool:
        """
        Whether model supports autoregressive generation.

        Returns:
            True - Mamba is a causal model
        """
        return True

    @classmethod
    def from_config(cls, config, **kwargs):
        """
        Create Mamba model from experiment configuration.

        Args:
            config: ExperimentConfig with model.architecture == "mamba"
            **kwargs: Additional arguments for model initialization

        Returns:
            MambaModel instance

        Raises:
            ValueError: If config doesn't specify Mamba architecture or mamba config
        """
        if config.model.architecture != "mamba":
            raise ValueError(
                f"MambaModel.from_config() called with architecture '{config.model.architecture}'. "
                "Expected 'mamba'."
            )

        if config.model.mamba is None:
            raise ValueError(
                "Mamba architecture requires 'mamba' configuration. "
                "Please specify model.mamba in your config."
            )

        # Extract Mamba configuration
        mamba_config = config.model.mamba

        # Get vocabulary size from tokenizer config
        vocab_size = config.tokenizer.vocab_size

        # Determine pad token ID (default to 0)
        pad_token_id = 0

        # Create Mamba model
        return cls(
            vocab_size=vocab_size,
            d_model=mamba_config.d_model,
            n_layers=mamba_config.n_layers,
            d_state=mamba_config.d_state,
            d_conv=mamba_config.d_conv,
            expand=mamba_config.expand,
            dropout=mamba_config.dropout,
            pad_token_id=pad_token_id,
            use_fast=kwargs.get('use_fast', True),
            **kwargs
        )
