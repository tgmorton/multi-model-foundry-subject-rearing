"""
Data collators for different language modeling objectives.

This module provides data collators for preparing batches during training:
- CausalLMDataCollator: For autoregressive language modeling (GPT-2, unidirectional LSTM)
- MaskedLMDataCollator: For masked language modeling (BERT, bidirectional models)

Data collators handle padding, label preparation, and objective-specific transformations.
"""

from typing import Dict, List, Any, Optional
import torch
from dataclasses import dataclass
import random


@dataclass
class CausalLMDataCollator:
    """
    Data collator for causal (autoregressive) language modeling.

    For causal LM, the labels are identical to the input_ids, shifted by the model
    internally. This collator handles padding and prepares batches for training.

    Attributes:
        tokenizer: Tokenizer with pad_token_id
        mlm: Always False for causal LM (kept for compatibility)
        pad_to_multiple_of: Pad sequence length to multiple of this value
    """

    tokenizer: Any
    mlm: bool = False
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Prepare a batch for causal language modeling.

        Args:
            examples: List of examples, each with 'input_ids' key

        Returns:
            Batch dictionary with 'input_ids', 'attention_mask', and 'labels'
        """
        # Extract input_ids from examples
        input_ids = [example['input_ids'] for example in examples]

        # Determine max length in batch
        max_length = max(len(ids) for ids in input_ids)

        # Pad to multiple if specified
        if self.pad_to_multiple_of is not None:
            max_length = (
                (max_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        # Pad sequences
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        padded_input_ids = []
        attention_masks = []

        for ids in input_ids:
            padding_length = max_length - len(ids)
            padded_ids = ids + [pad_token_id] * padding_length
            attention_mask = [1] * len(ids) + [0] * padding_length

            padded_input_ids.append(padded_ids)
            attention_masks.append(attention_mask)

        # Convert to tensors
        batch = {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
        }

        # For causal LM, labels are the same as input_ids
        # Padding tokens will be ignored in loss calculation (-100)
        labels = batch['input_ids'].clone()
        labels[batch['attention_mask'] == 0] = -100

        batch['labels'] = labels

        return batch


@dataclass
class MaskedLMDataCollator:
    """
    Data collator for masked language modeling (BERT-style).

    This collator randomly masks tokens in the input and creates labels for
    predicting the masked tokens. Follows the BERT masking strategy:
    - 80% of the time: replace with [MASK] token
    - 10% of the time: replace with random token
    - 10% of the time: keep original token

    Attributes:
        tokenizer: Tokenizer with mask_token_id
        mlm: Always True for masked LM
        mlm_probability: Probability of masking each token
        pad_to_multiple_of: Pad sequence length to multiple of this value
    """

    tokenizer: Any
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Prepare a batch for masked language modeling.

        Args:
            examples: List of examples, each with 'input_ids' key

        Returns:
            Batch dictionary with 'input_ids', 'attention_mask', and 'labels'
        """
        # Extract input_ids from examples
        input_ids = [example['input_ids'] for example in examples]

        # Determine max length in batch
        max_length = max(len(ids) for ids in input_ids)

        # Pad to multiple if specified
        if self.pad_to_multiple_of is not None:
            max_length = (
                (max_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        # Pad sequences
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        padded_input_ids = []
        attention_masks = []

        for ids in input_ids:
            padding_length = max_length - len(ids)
            padded_ids = ids + [pad_token_id] * padding_length
            attention_mask = [1] * len(ids) + [0] * padding_length

            padded_input_ids.append(padded_ids)
            attention_masks.append(attention_mask)

        # Convert to tensors
        batch = {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
        }

        # Apply masking for MLM
        if self.mlm:
            batch['input_ids'], batch['labels'] = self._mask_tokens(
                batch['input_ids'], batch['attention_mask']
            )
        else:
            # If MLM is disabled, labels are the same as input_ids
            labels = batch['input_ids'].clone()
            labels[batch['attention_mask'] == 0] = -100
            batch['labels'] = labels

        return batch

    def _mask_tokens(
        self, inputs: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling.

        Implements the BERT masking strategy:
        - 80% MASK token
        - 10% random token
        - 10% unchanged

        Args:
            inputs: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Tuple of (masked_inputs, labels)
        """
        labels = inputs.clone()

        # Create probability matrix for masking
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # Don't mask special tokens
        special_tokens_mask = self._get_special_tokens_mask(labels)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Don't mask padding tokens
        probability_matrix.masked_fill_(attention_mask == 0, value=0.0)

        # Sample which tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Only compute loss on masked tokens
        labels[~masked_indices] = -100

        # 80% of the time, replace masked input tokens with [MASK]
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # 10% of the time, keep masked input tokens unchanged (already done)

        return inputs, labels

    def _get_special_tokens_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Get mask indicating which tokens are special tokens.

        Args:
            token_ids: Token IDs [batch_size, seq_len]

        Returns:
            Boolean mask [batch_size, seq_len] where True = special token
        """
        special_tokens_mask = torch.zeros(token_ids.shape, dtype=torch.bool)

        # Check for each special token
        special_token_ids = []

        if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
            special_token_ids.append(self.tokenizer.bos_token_id)
        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
            special_token_ids.append(self.tokenizer.eos_token_id)
        if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
            special_token_ids.append(self.tokenizer.pad_token_id)
        if hasattr(self.tokenizer, 'cls_token_id') and self.tokenizer.cls_token_id is not None:
            special_token_ids.append(self.tokenizer.cls_token_id)
        if hasattr(self.tokenizer, 'sep_token_id') and self.tokenizer.sep_token_id is not None:
            special_token_ids.append(self.tokenizer.sep_token_id)
        if hasattr(self.tokenizer, 'unk_token_id') and self.tokenizer.unk_token_id is not None:
            special_token_ids.append(self.tokenizer.unk_token_id)

        # Mark positions of special tokens
        for special_id in special_token_ids:
            special_tokens_mask |= (token_ids == special_id)

        return special_tokens_mask


def get_data_collator(config, tokenizer):
    """
    Factory function to create appropriate data collator based on training objective.

    Args:
        config: ExperimentConfig with training.objective specified
        tokenizer: Tokenizer instance

    Returns:
        CausalLMDataCollator or MaskedLMDataCollator

    Raises:
        ValueError: If training objective is not recognized
    """
    objective = config.training.objective

    if objective == "causal_lm":
        return CausalLMDataCollator(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
    elif objective == "masked_lm":
        mlm_probability = getattr(config.training, 'mlm_probability', 0.15)
        return MaskedLMDataCollator(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
            pad_to_multiple_of=8
        )
    else:
        raise ValueError(
            f"Unknown training objective: '{objective}'. "
            "Must be 'causal_lm' or 'masked_lm'."
        )
