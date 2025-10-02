"""
Tokenizer factory for creating and training different tokenizer types.

This module provides a unified interface for creating tokenizers appropriate
for different model architectures:
- SentencePiece (Unigram): General purpose, used for GPT-2
- WordPiece: BERT-style tokenizer
- BPE: Byte-Pair Encoding tokenizer
- Character: Character-level tokenizer for RNNs

Each tokenizer type can be trained from scratch on the training corpus.
"""

import os
from pathlib import Path
from typing import Optional, Dict, List, Any
import glob
import json

# SentencePiece for Unigram tokenizer
import sentencepiece as spm

# HuggingFace tokenizers library
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence as NormalizerSequence
from transformers import PreTrainedTokenizerFast


class TokenizerFactory:
    """
    Factory for creating and training tokenizers for different architectures.

    Supported tokenizer types:
    - sentencepiece: Unigram LM tokenizer (default for GPT-2)
    - wordpiece: WordPiece tokenizer (BERT-style)
    - bpe: Byte-Pair Encoding tokenizer
    - character: Character-level tokenizer
    """

    @staticmethod
    def train_sentencepiece(
        input_files: List[str],
        output_dir: str,
        vocab_size: int,
        special_tokens: Optional[Dict[str, str]] = None
    ) -> PreTrainedTokenizerFast:
        """
        Train a SentencePiece Unigram tokenizer.

        Args:
            input_files: List of training corpus file paths
            output_dir: Directory to save tokenizer
            vocab_size: Target vocabulary size
            special_tokens: Special tokens dict (bos, eos, unk, pad)

        Returns:
            HuggingFace fast tokenizer
        """
        os.makedirs(output_dir, exist_ok=True)
        model_prefix = os.path.join(output_dir, 'tokenizer')

        # Default special tokens for GPT-2 style
        if special_tokens is None:
            special_tokens = {
                'bos_token': '<s>',
                'eos_token': '</s>',
                'unk_token': '<unk>',
                'pad_token': '<pad>'
            }

        # Train SentencePiece model
        training_input = ','.join(input_files)
        spm_args = {
            'input': training_input,
            'model_prefix': model_prefix,
            'vocab_size': vocab_size,
            'model_type': 'unigram',
            'max_sentence_length': 8192,
            'character_coverage': 1.0,
            'hard_vocab_limit': 'false',
        }
        arg_string = ' '.join([f'--{key}={value}' for key, value in spm_args.items()])

        print(f"  - Training SentencePiece tokenizer...")
        spm.SentencePieceTrainer.train(arg_string)

        # Convert to HuggingFace format
        from tokenizers import SentencePieceUnigramTokenizer

        sp_model_path = f"{model_prefix}.model"
        tokenizer_backend = SentencePieceUnigramTokenizer(sp_model_path)

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_backend,
            **special_tokens,
            model_max_length=8192
        )

        # Add special tokens
        tokenizer.add_special_tokens(special_tokens)

        # Save tokenizer
        tokenizer.save_pretrained(output_dir)
        print(f"  - SentencePiece tokenizer saved to: {output_dir}")

        return tokenizer

    @staticmethod
    def train_wordpiece(
        input_files: List[str],
        output_dir: str,
        vocab_size: int,
        special_tokens: Optional[Dict[str, str]] = None
    ) -> PreTrainedTokenizerFast:
        """
        Train a WordPiece tokenizer (BERT-style).

        Args:
            input_files: List of training corpus file paths
            output_dir: Directory to save tokenizer
            vocab_size: Target vocabulary size
            special_tokens: Special tokens dict (cls, sep, mask, unk, pad)

        Returns:
            HuggingFace fast tokenizer
        """
        os.makedirs(output_dir, exist_ok=True)

        # Default special tokens for BERT style
        if special_tokens is None:
            special_tokens = {
                'cls_token': '[CLS]',
                'sep_token': '[SEP]',
                'mask_token': '[MASK]',
                'unk_token': '[UNK]',
                'pad_token': '[PAD]'
            }

        # Create WordPiece tokenizer
        tokenizer = Tokenizer(models.WordPiece(unk_token=special_tokens['unk_token']))

        # Set normalizer (lowercase + NFD + strip accents like BERT)
        tokenizer.normalizer = NormalizerSequence([NFD(), Lowercase(), StripAccents()])

        # Set pre-tokenizer (split on whitespace and punctuation)
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        # Create trainer
        special_token_list = list(special_tokens.values())
        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=special_token_list,
            min_frequency=2,
            continuing_subword_prefix='##'
        )

        # Train tokenizer
        print(f"  - Training WordPiece tokenizer on {len(input_files)} files...")
        tokenizer.train(input_files, trainer)

        # Add post-processor for BERT format ([CLS] ... [SEP])
        cls_token_id = tokenizer.token_to_id(special_tokens['cls_token'])
        sep_token_id = tokenizer.token_to_id(special_tokens['sep_token'])

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{special_tokens['cls_token']} $A {special_tokens['sep_token']}",
            pair=f"{special_tokens['cls_token']} $A {special_tokens['sep_token']} $B:1 {special_tokens['sep_token']}:1",
            special_tokens=[
                (special_tokens['cls_token'], cls_token_id),
                (special_tokens['sep_token'], sep_token_id),
            ],
        )

        # Wrap in PreTrainedTokenizerFast
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            **special_tokens,
            model_max_length=512  # BERT default
        )

        # Save tokenizer
        fast_tokenizer.save_pretrained(output_dir)
        print(f"  - WordPiece tokenizer saved to: {output_dir}")

        return fast_tokenizer

    @staticmethod
    def train_bpe(
        input_files: List[str],
        output_dir: str,
        vocab_size: int,
        special_tokens: Optional[Dict[str, str]] = None
    ) -> PreTrainedTokenizerFast:
        """
        Train a BPE (Byte-Pair Encoding) tokenizer.

        Args:
            input_files: List of training corpus file paths
            output_dir: Directory to save tokenizer
            vocab_size: Target vocabulary size
            special_tokens: Special tokens dict

        Returns:
            HuggingFace fast tokenizer
        """
        os.makedirs(output_dir, exist_ok=True)

        # Default special tokens
        if special_tokens is None:
            special_tokens = {
                'bos_token': '<s>',
                'eos_token': '</s>',
                'unk_token': '<unk>',
                'pad_token': '<pad>'
            }

        # Create BPE tokenizer
        tokenizer = Tokenizer(models.BPE(unk_token=special_tokens['unk_token']))

        # Set pre-tokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # Create trainer
        special_token_list = list(special_tokens.values())
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_token_list,
            min_frequency=2
        )

        # Train tokenizer
        print(f"  - Training BPE tokenizer on {len(input_files)} files...")
        tokenizer.train(input_files, trainer)

        # Wrap in PreTrainedTokenizerFast
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            **special_tokens,
            model_max_length=8192
        )

        # Save tokenizer
        fast_tokenizer.save_pretrained(output_dir)
        print(f"  - BPE tokenizer saved to: {output_dir}")

        return fast_tokenizer

    @staticmethod
    def train_character(
        input_files: List[str],
        output_dir: str,
        special_tokens: Optional[Dict[str, str]] = None
    ) -> PreTrainedTokenizerFast:
        """
        Create a character-level tokenizer.

        For character tokenizers, we build vocabulary from all unique characters
        in the training corpus rather than using a fixed vocab_size.

        Args:
            input_files: List of training corpus file paths
            output_dir: Directory to save tokenizer
            special_tokens: Special tokens dict

        Returns:
            HuggingFace fast tokenizer
        """
        os.makedirs(output_dir, exist_ok=True)

        # Default special tokens
        if special_tokens is None:
            special_tokens = {
                'bos_token': '<s>',
                'eos_token': '</s>',
                'unk_token': '<unk>',
                'pad_token': '<pad>'
            }

        # Collect all unique characters from training files
        print(f"  - Building character vocabulary from {len(input_files)} files...")
        chars = set()
        for file_path in input_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    chars.update(line)

        # Build vocabulary
        vocab = list(special_tokens.values()) + sorted(list(chars))
        vocab_dict = {token: idx for idx, token in enumerate(vocab)}

        print(f"  - Character vocabulary size: {len(vocab)}")

        # Save vocabulary
        vocab_path = os.path.join(output_dir, 'vocab.json')
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

        # Create tokenizer config
        tokenizer_config = {
            'tokenizer_class': 'PreTrainedTokenizerFast',
            'model_max_length': 8192,
            'padding_side': 'right',
            'truncation_side': 'right',
            **{f'{k}': v for k, v in special_tokens.items()}
        }

        config_path = os.path.join(output_dir, 'tokenizer_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, indent=2)

        print(f"  - Character tokenizer saved to: {output_dir}")
        print(f"  - Note: Character tokenizer requires custom loading logic")

        # For now, return None as character tokenizers need special handling
        # This will be improved in future iterations
        return None

    @staticmethod
    def train_tokenizer(
        tokenizer_type: str,
        input_files: List[str],
        output_dir: str,
        vocab_size: int,
        special_tokens: Optional[Dict[str, str]] = None
    ) -> Optional[PreTrainedTokenizerFast]:
        """
        Train a tokenizer of the specified type.

        Args:
            tokenizer_type: Type of tokenizer ('sentencepiece', 'wordpiece', 'bpe', 'character')
            input_files: List of training corpus file paths
            output_dir: Directory to save tokenizer
            vocab_size: Target vocabulary size (ignored for character tokenizer)
            special_tokens: Special tokens dict (architecture-specific)

        Returns:
            Trained tokenizer or None if tokenizer type requires special handling

        Raises:
            ValueError: If tokenizer type is not supported
        """
        if tokenizer_type == 'sentencepiece':
            return TokenizerFactory.train_sentencepiece(
                input_files, output_dir, vocab_size, special_tokens
            )
        elif tokenizer_type == 'wordpiece':
            return TokenizerFactory.train_wordpiece(
                input_files, output_dir, vocab_size, special_tokens
            )
        elif tokenizer_type == 'bpe':
            return TokenizerFactory.train_bpe(
                input_files, output_dir, vocab_size, special_tokens
            )
        elif tokenizer_type == 'character':
            return TokenizerFactory.train_character(
                input_files, output_dir, special_tokens
            )
        else:
            raise ValueError(
                f"Unknown tokenizer type: '{tokenizer_type}'. "
                "Supported types: sentencepiece, wordpiece, bpe, character"
            )


def train_tokenizer_from_config(config_path: str, project_root: Optional[str] = None):
    """
    Train a tokenizer using parameters from a YAML experiment config file.

    This function replaces the standalone train_tokenizer.py script functionality
    with support for multiple tokenizer types.

    Args:
        config_path: Path to YAML configuration file
        project_root: Project root directory (auto-detected if None)

    Returns:
        Trained tokenizer instance
    """
    import yaml

    if project_root is None:
        project_root = _find_project_root(config_path)

    print(f"--- Training Tokenizer from Config: {config_path} ---")
    print(f"  - Project Root: {project_root}")

    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Extract paths
    training_corpus_path_from_config = config['data']['training_corpus']
    output_dir_from_config = config['tokenizer']['output_dir']

    training_corpus_path = (
        training_corpus_path_from_config
        if os.path.isabs(training_corpus_path_from_config)
        else os.path.join(project_root, training_corpus_path_from_config)
    )
    output_dir = (
        output_dir_from_config
        if os.path.isabs(output_dir_from_config)
        else os.path.join(project_root, output_dir_from_config)
    )

    # Extract tokenizer parameters
    vocab_size = config['tokenizer']['vocab_size']
    tokenizer_type = config['tokenizer'].get('tokenizer_type', 'sentencepiece')
    special_tokens = config['tokenizer'].get('special_tokens', None)
    experiment_name = config.get('experiment_name', 'tokenizer')

    # Find input files
    if not os.path.exists(training_corpus_path):
        raise FileNotFoundError(f"Training corpus not found at: {training_corpus_path}")

    if os.path.isdir(training_corpus_path):
        print(f"  - Searching for training files in: {training_corpus_path}")
        train_files = glob.glob(os.path.join(training_corpus_path, '**', '*.train'), recursive=True)
        test_files = glob.glob(os.path.join(training_corpus_path, '**', '*.test'), recursive=True)
        input_files = train_files + test_files

        if not input_files:
            raise FileNotFoundError(f"No .train or .test files found in: {training_corpus_path}")

        print(f"  - Found {len(input_files)} files for tokenizer training")
    else:
        input_files = [training_corpus_path]
        print(f"  - Using single file: {training_corpus_path}")

    print(f"  - Tokenizer Type: {tokenizer_type}")
    print(f"  - Vocab Size: {vocab_size}")
    print(f"  - Output Dir: {output_dir}")

    # Train tokenizer
    tokenizer = TokenizerFactory.train_tokenizer(
        tokenizer_type=tokenizer_type,
        input_files=input_files,
        output_dir=output_dir,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )

    print(f"  - Successfully trained {tokenizer_type} tokenizer for '{experiment_name}'")
    print("----- Tokenizer Training Complete -----")

    return tokenizer


def _find_project_root(start_path: str) -> str:
    """Find project root by searching for .git directory."""
    path = Path(start_path).resolve()
    while path.parent != path:
        if (path / '.git').is_dir():
            return str(path)
        path = path.parent
    print("Warning: .git directory not found. Using current working directory.")
    return os.getcwd()
