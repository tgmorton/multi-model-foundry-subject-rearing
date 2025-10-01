"""
Tokenizer loading and wrapping utilities.

This module handles loading of various tokenizer types including SentencePiece
models and provides wrapper classes for compatibility with Hugging Face APIs.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Union, List, Dict, Any
import sentencepiece as spm
import torch
from transformers import AutoTokenizer


def load_tokenizer(tokenizer_path: str):
    """
    Load a SentencePiece tokenizer and wrap it for Hugging Face compatibility.

    Args:
        tokenizer_path: Path to the tokenizer directory

    Returns:
        A tokenizer instance compatible with Hugging Face APIs
    """
    try:
        # First try to load as a standard Hugging Face tokenizer
        return AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    except Exception as e:
        print(f"  - Standard tokenizer loading failed: {e}")
        print("  - Attempting to load SentencePiece tokenizer directly...")

        # Load SentencePiece model directly
        sp_model_path = os.path.join(tokenizer_path, "tokenizer.model")
        if not os.path.exists(sp_model_path):
            raise FileNotFoundError(f"SentencePiece model not found at {sp_model_path}")

        # Create a simple wrapper tokenizer
        sp_processor = spm.SentencePieceProcessor()
        sp_processor.load(sp_model_path)

        print("  - Successfully loaded SentencePiece tokenizer with wrapper")
        return SentencePieceTokenizerWrapper(sp_processor, tokenizer_path)


class SentencePieceTokenizerWrapper:
    """
    Wrapper class for SentencePiece tokenizers to provide Hugging Face API compatibility.

    This wrapper provides the standard tokenizer interface expected by Hugging Face
    models and training utilities.
    """

    def __init__(self, sp_processor: spm.SentencePieceProcessor, tokenizer_path: str):
        """
        Initialize the wrapper.

        Args:
            sp_processor: Loaded SentencePiece processor
            tokenizer_path: Path to the tokenizer directory (for saving)
        """
        self.sp_processor = sp_processor
        self.tokenizer_path = tokenizer_path
        self.vocab_size = sp_processor.vocab_size()
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.pad_token_id = sp_processor.piece_to_id("<pad>")
        self.eos_token_id = sp_processor.piece_to_id("</s>")
        self.unk_token_id = sp_processor.piece_to_id("<unk>")
        self.bos_token_id = sp_processor.piece_to_id("<s>")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token ids.

        Args:
            text: Input text to encode
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        if add_special_tokens:
            return [self.bos_token_id] + self.sp_processor.encode(text) + [self.eos_token_id]
        else:
            return self.sp_processor.encode(text)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token ids to text.

        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text string
        """
        if skip_special_tokens:
            # Filter out special tokens
            special_tokens = {self.pad_token_id, self.eos_token_id, self.unk_token_id, self.bos_token_id}
            token_ids = [tid for tid in token_ids if tid not in special_tokens]
        return self.sp_processor.decode(token_ids)

    def save_pretrained(self, save_directory: Union[str, Path]):
        """
        Save tokenizer to directory (SentencePiece compatible).

        Args:
            save_directory: Directory to save the tokenizer files
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Copy the SentencePiece model file
        tokenizer_dir = Path(self.tokenizer_path)
        sp_model_path = tokenizer_dir / "tokenizer.model"
        if sp_model_path.exists():
            shutil.copy2(sp_model_path, save_directory / "tokenizer.model")

        # Copy config files if they exist
        for config_file in ["tokenizer_config.json", "special_tokens_map.json"]:
            config_path = tokenizer_dir / config_file
            if config_path.exists():
                shutil.copy2(config_path, save_directory / config_file)

        # Create a simple config if none exists
        config_path = save_directory / "tokenizer_config.json"
        if not config_path.exists():
            config = {
                "tokenizer_type": "sentencepiece",
                "vocab_size": self.vocab_size,
                "pad_token": self.pad_token,
                "eos_token": self.eos_token,
                "unk_token": self.unk_token,
                "bos_token": self.bos_token
            }
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

    def __call__(self, text: Union[str, List[str]], padding: bool = False,
                 truncation: bool = False, max_length: int = None,
                 return_tensors: str = None) -> Dict[str, Any]:
        """
        Tokenize text (basic implementation).

        Args:
            text: Input text or list of texts
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            return_tensors: Format for return ('pt' for PyTorch tensors)

        Returns:
            Dictionary with 'input_ids' key
        """
        if isinstance(text, str):
            text = [text]

        encoded = []
        for t in text:
            tokens = self.encode(t, add_special_tokens=True)
            if truncation and max_length and len(tokens) > max_length:
                tokens = tokens[:max_length]
            encoded.append(tokens)

        if padding:
            max_len = max(len(tokens) for tokens in encoded) if not max_length else max_length
            for tokens in encoded:
                while len(tokens) < max_len:
                    tokens.append(self.pad_token_id)

        if return_tensors == "pt":
            return {"input_ids": torch.tensor(encoded)}
        else:
            return {"input_ids": encoded}

    def pad(self, encoded_inputs: Union[Dict, List], padding: bool = True,
            max_length: int = None, pad_to_multiple_of: int = None,
            return_tensors: str = None) -> Dict[str, Any]:
        """
        Pad sequences to the same length.

        Args:
            encoded_inputs: Input sequences to pad
            padding: Whether to apply padding
            max_length: Maximum length after padding
            pad_to_multiple_of: Pad to multiple of this value
            return_tensors: Format for return ('pt' for PyTorch tensors)

        Returns:
            Dictionary with 'input_ids' and 'attention_mask' keys
        """
        # Handle different input formats
        if isinstance(encoded_inputs, dict):
            if "input_ids" in encoded_inputs:
                input_ids = encoded_inputs["input_ids"]
            else:
                # If it's a dict but no input_ids, assume it's a single example
                input_ids = [encoded_inputs]
        elif isinstance(encoded_inputs, list):
            # Check if it's a list of dicts (batch) or list of lists (sequences)
            if encoded_inputs and isinstance(encoded_inputs[0], dict):
                # List of dicts - extract input_ids from each
                input_ids = []
                for item in encoded_inputs:
                    if "input_ids" in item:
                        input_ids.append(item["input_ids"])
                    else:
                        # Assume the dict itself contains the sequence
                        input_ids.append(list(item.values())[0] if item else [])
            else:
                # List of sequences or single sequence
                input_ids = encoded_inputs
        else:
            input_ids = encoded_inputs

        # Ensure input_ids is a list of lists
        if input_ids and not isinstance(input_ids[0], (list, tuple)):
            input_ids = [input_ids]

        # Handle empty input
        if not input_ids:
            return {"input_ids": [], "attention_mask": []}

        # Determine max length
        if max_length is None:
            max_length = max(len(seq) for seq in input_ids if seq)

        # Pad sequences
        padded_input_ids = []
        attention_mask = []

        for seq in input_ids:
            # Ensure seq is a list
            if isinstance(seq, dict):
                if "input_ids" in seq:
                    seq = seq["input_ids"]
                else:
                    seq = list(seq.values())[0] if seq else []
            elif not isinstance(seq, (list, tuple)):
                seq = [seq]

            # Truncate if necessary
            if len(seq) > max_length:
                seq = seq[:max_length]

            # Create attention mask (1 for real tokens, 0 for padding)
            mask = [1] * len(seq) + [0] * (max_length - len(seq))
            attention_mask.append(mask)

            # Pad sequence
            padded_seq = list(seq) + [self.pad_token_id] * (max_length - len(seq))
            padded_input_ids.append(padded_seq)

        result = {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask
        }

        # Convert to tensors if requested
        if return_tensors == "pt":
            result = {k: torch.tensor(v) for k, v in result.items()}

        return result
