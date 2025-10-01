"""
Model and tokenizer loading utilities for evaluation.
Compatible with model-foundry trained checkpoints.
"""

import os
import torch
from pathlib import Path
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoConfig
import sentencepiece as spm
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading of model checkpoints and tokenizers for evaluation."""
    
    def __init__(self, device: str = None):
        """
        Initialize the model loader.
        
        Args:
            device: Device to load model on ('cuda', 'cpu', or None for auto-detect)
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        logger.info(f"ModelLoader initialized with device: {self.device}")
    
    def load_checkpoint(
        self, 
        checkpoint_path: str,
        use_fp16: bool = False,
        load_in_8bit: bool = False
    ) -> AutoModelForCausalLM:
        """
        Load a model checkpoint optimized for inference.
        
        Args:
            checkpoint_path: Path to the checkpoint directory
            use_fp16: Whether to use mixed precision for faster inference
            load_in_8bit: Whether to use 8-bit quantization (requires bitsandbytes)
            
        Returns:
            Loaded model ready for evaluation
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading model from {checkpoint_path}")
        
        # Load model configuration
        config_path = checkpoint_path / "config.json"
        if config_path.exists():
            config = AutoConfig.from_pretrained(checkpoint_path)
        else:
            # Fallback to GPT-2 config if not found
            logger.warning("Config not found, using default GPT-2 configuration")
            config = AutoConfig.from_pretrained("gpt2")
        
        # Set inference-specific configurations
        config.use_cache = True  # Enable KV cache for faster generation
        config.output_hidden_states = False
        config.output_attentions = False
        
        # Load model with optimizations
        load_kwargs = {
            "config": config,
            "torch_dtype": torch.float16 if use_fp16 else torch.float32,
        }
        
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
            
        try:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                **load_kwargs
            )
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            # Try loading just the pytorch_model.bin
            model = AutoModelForCausalLM.from_config(config)
            state_dict_path = checkpoint_path / "pytorch_model.bin"
            if state_dict_path.exists():
                state_dict = torch.load(state_dict_path, map_location=self.device)
                model.load_state_dict(state_dict)
            else:
                raise RuntimeError(f"Could not find model weights in {checkpoint_path}")
        
        # Move to device and set to eval mode
        if not load_in_8bit:  # 8-bit models handle device placement internally
            model = model.to(self.device)
        model.eval()
        
        # Log model info
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded with {param_count:,} parameters")
        logger.info(f"Using dtype: {next(model.parameters()).dtype}")
        
        return model
    
    def load_tokenizer(self, tokenizer_path: str) -> spm.SentencePieceProcessor:
        """
        Load a SentencePiece tokenizer.
        
        Args:
            tokenizer_path: Path to tokenizer directory or model file
            
        Returns:
            Loaded SentencePiece tokenizer
        """
        tokenizer_path = Path(tokenizer_path)
        
        # Check if path is directory or file
        if tokenizer_path.is_dir():
            model_file = tokenizer_path / "tokenizer.model"
        else:
            model_file = tokenizer_path
            
        if not model_file.exists():
            raise FileNotFoundError(f"Tokenizer model not found: {model_file}")
        
        logger.info(f"Loading tokenizer from {model_file}")
        
        # Load SentencePiece model
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(str(model_file))
        
        logger.info(f"Tokenizer loaded with vocab size: {tokenizer.vocab_size()}")
        
        return tokenizer
    
    def load_model_and_tokenizer(
        self,
        checkpoint_path: str,
        tokenizer_path: str,
        use_fp16: bool = False
    ) -> Tuple[AutoModelForCausalLM, spm.SentencePieceProcessor]:
        """
        Convenience method to load both model and tokenizer.
        
        Args:
            checkpoint_path: Path to model checkpoint
            tokenizer_path: Path to tokenizer
            use_fp16: Whether to use mixed precision
            
        Returns:
            Tuple of (model, tokenizer)
        """
        model = self.load_checkpoint(checkpoint_path, use_fp16=use_fp16)
        tokenizer = self.load_tokenizer(tokenizer_path)
        
        # Verify compatibility
        if hasattr(model.config, 'vocab_size'):
            model_vocab = model.config.vocab_size
            tokenizer_vocab = tokenizer.vocab_size()
            if model_vocab != tokenizer_vocab:
                logger.warning(
                    f"Vocab size mismatch: model={model_vocab}, "
                    f"tokenizer={tokenizer_vocab}"
                )
        
        return model, tokenizer
    
    def find_checkpoints(self, experiment_dir: str) -> list:
        """
        Find all checkpoints in an experiment directory.
        
        Args:
            experiment_dir: Path to experiment directory
            
        Returns:
            List of checkpoint paths sorted by step number
        """
        experiment_dir = Path(experiment_dir)
        
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
        
        # Look for checkpoint directories
        checkpoints = []
        for item in experiment_dir.iterdir():
            if item.is_dir() and (
                item.name.startswith("checkpoint-") or 
                item.name.startswith("epoch_")
            ):
                checkpoints.append(item)
        
        # Sort by step/epoch number
        def get_checkpoint_number(path):
            name = path.name
            if "checkpoint-" in name:
                return int(name.split("-")[1])
            elif "epoch_" in name:
                return int(name.split("_")[1]) * 10000  # Arbitrary large multiplier
            return 0
        
        checkpoints.sort(key=get_checkpoint_number)
        
        logger.info(f"Found {len(checkpoints)} checkpoints in {experiment_dir}")
        
        return checkpoints


# Utility functions for memory management
def clear_gpu_cache():
    """Clear GPU memory cache if using CUDA."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_model_memory_usage(model: torch.nn.Module) -> dict:
    """
    Get memory usage statistics for a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with memory statistics
    """
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
    
    stats = {
        "param_memory_mb": param_memory / 1024 / 1024,
        "buffer_memory_mb": buffer_memory / 1024 / 1024,
        "total_memory_mb": (param_memory + buffer_memory) / 1024 / 1024,
    }
    
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        stats["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        stats["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
    
    return stats