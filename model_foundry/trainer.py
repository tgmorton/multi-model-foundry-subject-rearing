import argparse
import os
import yaml
import glob
import re
import logging
import time
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler, AutoTokenizer, PreTrainedTokenizerFast
import sentencepiece as spm
from tqdm.auto import tqdm
import wandb
import random
import numpy as np
import subprocess

# Import the new, refactored components
from .config import ExperimentConfig
from .model import create_model
from .utils import find_project_root, set_seed, get_device, get_git_commit_hash
from .data import create_data_processor
from .logging_utils import setup_logging


# Removed _chunk_examples function - now handled by DataProcessor


class Trainer:
    def __init__(self, config: ExperimentConfig, base_dir: str):
        self.config = config
        self.base_dir = base_dir
        self.device = get_device()
        self.git_commit_hash = get_git_commit_hash()

        # State variables to be initialized
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.dataloader = None
        self.tokenizer = None
        self.global_step = 0
        self.epoch = 0
        
        # Enhanced AMP setup
        self.scaler = None
        self.amp_enabled = False
        if self.config.training.use_amp and torch.cuda.is_available():
            self.amp_enabled = True
            # Use higher initial scale for better stability
            self.scaler = torch.cuda.amp.GradScaler(
                init_scale=2**16,
                growth_factor=2,
                backoff_factor=0.5,
                growth_interval=2000,
                enabled=True
            )
            print("  - AMP enabled with enhanced GradScaler settings")
            
        # Memory management settings
        if torch.cuda.is_available():
            # Set memory fraction to prevent OOM
            torch.cuda.set_per_process_memory_fraction(0.95)  # Use max 95% of GPU memory
            
            # Configure allocator for better performance
            # Respect external setting if provided; otherwise set a conservative default
            if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ or not os.environ['PYTORCH_CUDA_ALLOC_CONF']:
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            
            # Enable cudnn benchmarking for consistent memory usage
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False  # Faster, less memory
            
            print("  - CUDA memory management configured (95% memory limit, max_split_size_mb:512)")
        
        # Initialize data processor
        self.data_processor = create_data_processor(config, base_dir)
        


    def _calculate_training_parameters(self):
        """Calculate training parameters based on dataset size if not explicitly set."""
        # Get steps per epoch from dataset
        try:
            steps_per_epoch = self.data_processor.get_training_steps_per_epoch()
        except:
            # Fallback: estimate based on typical dataset size if preprocessing hasn't run yet
            print("  ⚠️  Could not determine exact dataset size, using estimates")
            steps_per_epoch = 100  # Conservative estimate
        
        # Calculate train_steps if not specified
        if self.config.training.train_steps is None:
            calculated_train_steps = steps_per_epoch * self.config.training.epochs
            print(f"  - Auto-calculated train_steps: {calculated_train_steps} ({self.config.training.epochs} epochs × {steps_per_epoch} steps/epoch)")
            # Update the config object
            self.config.training.train_steps = calculated_train_steps
        
        # Calculate warmup_steps if not specified
        if self.config.training.warmup_steps is None:
            calculated_warmup_steps = int(self.config.training.train_steps * self.config.training.warmup_ratio)
            print(f"  - Auto-calculated warmup_steps: {calculated_warmup_steps} ({self.config.training.warmup_ratio:.1%} of total steps)")
            # Update the config object
            self.config.training.warmup_steps = calculated_warmup_steps
        
        print(f"  - Final training configuration:")
        print(f"    - Epochs: {self.config.training.epochs}")
        print(f"    - Steps per epoch: {steps_per_epoch}")
        print(f"    - Total training steps: {self.config.training.train_steps}")
        print(f"    - Warmup steps: {self.config.training.warmup_steps}")

    def _prepare_data(self):
        """Loads and prepares the dataset and dataloader."""
        # Preprocess data into fixed-length chunks
        if not self.data_processor.preprocess_data():
            raise RuntimeError("Data preprocessing failed")
        
        # Create dataloader
        self.dataloader = self.data_processor.create_dataloader(self.tokenizer)

    def _get_checkpoint_schedule(self) -> set:
        """
        Get the checkpoint schedule, either from config or by generating it dynamically.
        
        Returns:
            Set of training steps at which to save checkpoints
        """
        # If auto-generation is enabled and no schedule exists, generate one
        if (self.config.training.auto_generate_checkpoints and 
            not self.config.training.checkpoint_schedule):
            
            print("  - Auto-generating checkpoint schedule...")
            
            # Import the schedule generation function
            from scripts.generate_checkpoint_schedule import (
                generate_checkpoint_schedule, 
                CheckpointGenerationConfig
            )
            
            # Create generation config
            generation_config = CheckpointGenerationConfig(
                first_epoch_checkpoints=self.config.training.first_epoch_checkpoints,
                subsequent_epochs_spacing=self.config.training.subsequent_epochs_spacing,
                log_base=self.config.training.log_base,
                linear_interval=self.config.training.linear_interval,
                min_interval=self.config.training.min_checkpoint_interval,
                min_checkpoints_per_epoch=self.config.training.min_checkpoints_per_epoch
            )
            
            # Generate schedule
            schedule = generate_checkpoint_schedule(
                self.config, 
                self.base_dir, 
                generation_config
            )
            
            # Update the config with the generated schedule
            self.config.training.checkpoint_schedule = schedule
            
            print(f"  - Generated {len(schedule)} checkpoint steps")
        
        # Return as set for efficient lookup
        return set(self.config.training.checkpoint_schedule or [])

    def _save_checkpoint(self):
        """Saves the complete training state to a checkpoint directory."""
        import datetime
        import hashlib
        import json
        
        checkpoint_dir = Path(self.base_dir) / self.config.training.output_dir / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save training state
        state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_random_state': torch.get_rng_state(),
            'torch_cuda_random_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'git_commit_hash': self.git_commit_hash,
            # Add AMP scaler state
            'amp_scaler': self.scaler.state_dict() if self.scaler is not None else None,
        }
        torch.save(state, checkpoint_dir / "training_state.pt")
        
        # Save checkpoint metadata
        metadata = {
            'experiment_name': self.config.experiment_name,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'timestamp': datetime.datetime.now().isoformat(),
            'git_commit_hash': self.git_commit_hash,
            'config_hash': hashlib.md5(json.dumps(self.config.model_dump(), sort_keys=True).encode()).hexdigest(),
            'wandb_run_id': wandb.run.id if self.config.logging.use_wandb and wandb.run else None,
            'training_config': {
                'learning_rate': self.config.training.learning_rate,
                'batch_size': self.config.data.batch_size,
                'gradient_accumulation_steps': self.config.training.gradient_accumulation_steps,
                'use_amp': self.config.training.use_amp,
                'use_tf32': self.config.training.use_tf32,
                'use_gradient_checkpointing': self.config.training.use_gradient_checkpointing,
            },
            'model_config': {
                'layers': self.config.model.layers,
                'hidden_size': self.config.model.hidden_size,
                'attention_heads': self.config.model.attention_heads,
            }
        }
        
        with open(checkpoint_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n  - Saved checkpoint at step {self.global_step} to '{checkpoint_dir}'")
        print(f"    - Config hash: {metadata['config_hash'][:8]}...")
        if metadata['wandb_run_id']:
            print(f"    - WandB run ID: {metadata['wandb_run_id']}")

    def _load_checkpoint(self):
        """Loads training state from the latest checkpoint if resume is enabled."""
        output_dir = Path(self.base_dir) / self.config.training.output_dir
        if not self.config.training.resume_from_checkpoint or not output_dir.exists():
            return

        checkpoints = glob.glob(str(output_dir / "checkpoint-*"))
        if not checkpoints:
            print("  - `resume_from_checkpoint` is true, but no checkpoints found. Starting fresh.")
            return

        # Find the checkpoint with the highest step number
        latest_checkpoint = max(checkpoints, key=lambda p: int(re.search(r'checkpoint-(\d+)', p).group(1)))
        print(f"  - Resuming training from latest checkpoint: {latest_checkpoint}")

        # Load tokenizer first as it's needed for model setup
        self.tokenizer = AutoTokenizer.from_pretrained(latest_checkpoint)

        # Load model and move to device
        self.model = create_model(self.config).to(self.device)
        self.model.load_state_dict(torch.load(Path(latest_checkpoint) / "pytorch_model.bin", map_location=self.device))

        # Load training state
        state = torch.load(Path(latest_checkpoint) / "training_state.pt", map_location="cpu")
        self.global_step = state['global_step']
        self.epoch = state['epoch']

        # Restore optimizer and scheduler states
        self.optimizer.load_state_dict(state['optimizer'])
        self.lr_scheduler.load_state_dict(state['lr_scheduler'])

        # Restore RNG states
        random.setstate(state['random_state'])
        np.random.set_state(state['numpy_random_state'])
        torch.set_rng_state(state['torch_random_state'])
        if torch.cuda.is_available() and state['torch_cuda_random_state']:
            torch.cuda.set_rng_state_all(state['torch_cuda_random_state'])

        # Restore AMP scaler state
        if self.scaler is not None and state.get('amp_scaler') is not None:
            self.scaler.load_state_dict(state['amp_scaler'])
            print(f"  - Restored AMP scaler state")

        print(f"  - Resumed from step {self.global_step} at epoch {self.epoch}.")

    def _load_sentencepiece_tokenizer(self, tokenizer_path: str):
        """Load a SentencePiece tokenizer and wrap it for Hugging Face compatibility."""
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
            
            # Create a basic tokenizer wrapper
            class SentencePieceTokenizerWrapper:
                def __init__(self, sp_processor):
                    self.sp_processor = sp_processor
                    self.vocab_size = sp_processor.vocab_size()
                    self.pad_token = "<pad>"
                    self.eos_token = "</s>"
                    self.unk_token = "<unk>"
                    self.bos_token = "<s>"
                    self.pad_token_id = sp_processor.piece_to_id("<pad>")
                    self.eos_token_id = sp_processor.piece_to_id("</s>")
                    self.unk_token_id = sp_processor.piece_to_id("<unk>")
                    self.bos_token_id = sp_processor.piece_to_id("<s>")
                
                def encode(self, text, add_special_tokens=True):
                    """Encode text to token ids."""
                    if add_special_tokens:
                        return [self.bos_token_id] + self.sp_processor.encode(text) + [self.eos_token_id]
                    else:
                        return self.sp_processor.encode(text)
                
                def decode(self, token_ids, skip_special_tokens=True):
                    """Decode token ids to text."""
                    if skip_special_tokens:
                        # Filter out special tokens
                        special_tokens = {self.pad_token_id, self.eos_token_id, self.unk_token_id, self.bos_token_id}
                        token_ids = [tid for tid in token_ids if tid not in special_tokens]
                    return self.sp_processor.decode(token_ids)
                
                def save_pretrained(self, save_directory):
                    """Save tokenizer to directory (SentencePiece compatible)."""
                    import os
                    import shutil
                    import json
                    
                    save_directory = Path(save_directory)
                    save_directory.mkdir(parents=True, exist_ok=True)
                    
                    # Copy the SentencePiece model file
                    tokenizer_dir = Path(tokenizer_path)
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
                
                def __call__(self, text, padding=False, truncation=False, max_length=None, return_tensors=None):
                    """Tokenize text (basic implementation)."""
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
                        import torch
                        return {"input_ids": torch.tensor(encoded)}
                    else:
                        return {"input_ids": encoded}
                
                def pad(self, encoded_inputs, padding=True, max_length=None, pad_to_multiple_of=None, return_tensors=None):
                    """Pad sequences to the same length."""
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
                        import torch
                        result = {k: torch.tensor(v) for k, v in result.items()}
                    
                    return result
            
            print("  - Successfully loaded SentencePiece tokenizer with wrapper")
            return SentencePieceTokenizerWrapper(sp_processor)

    def _save_environment_snapshot(self):
        """Save environment snapshot for reproducibility."""
        import datetime
        import subprocess
        
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(self.base_dir, self.config.logging.dir, self.config.experiment_name)
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        env_file = os.path.join(log_dir, f"env_{timestamp}.txt")
        
        try:
            with open(env_file, 'w') as f:
                f.write(f"Experiment: {self.config.experiment_name}\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                f.write(f"Git commit: {self.git_commit_hash}\n\n")
                
                # Save pip freeze
                f.write("=== PIP PACKAGES ===\n")
                try:
                    pip_output = subprocess.check_output(['pip', 'freeze'], text=True)
                    f.write(pip_output)
                except Exception as e:
                    f.write(f"Error getting pip packages: {e}\n")
                
                f.write("\n=== CUDA INFO ===\n")
                if torch.cuda.is_available():
                    f.write(f"CUDA available: True\n")
                    f.write(f"CUDA version: {torch.version.cuda}\n")
                    f.write(f"GPU count: {torch.cuda.device_count()}\n")
                    for i in range(torch.cuda.device_count()):
                        f.write(f"GPU {i}: {torch.cuda.get_device_name(i)}\n")
                    
                    # Get CUDA driver info
                    try:
                        nvidia_output = subprocess.check_output(['nvidia-smi'], text=True)
                        f.write(f"\nNVIDIA-SMI output:\n{nvidia_output}\n")
                    except Exception as e:
                        f.write(f"Error getting NVIDIA-SMI info: {e}\n")
                else:
                    f.write("CUDA available: False\n")
                
                f.write(f"\n=== SYSTEM INFO ===\n")
                f.write(f"PyTorch version: {torch.__version__}\n")
                f.write(f"Python version: {subprocess.check_output(['python', '--version'], text=True).strip()}\n")
                
            print(f"  - Environment snapshot saved to: {env_file}")
            
        except Exception as e:
            print(f"  - Warning: Failed to save environment snapshot: {e}")

    def train(self):
        """Main training loop."""
        # Set up unified logging
        logger = setup_logging("trainer", experiment=self.config.experiment_name, 
                              log_dir=self.config.logging.dir,
                              level=getattr(logging, self.config.logging.level))
        logger.info(f"--- Starting Training Run for: {self.config.experiment_name} ---")
        
        try:
            return self._train_loop()
        except Exception as e:
            # Log the error without verbose traceback
            logger.error(f"Training failed: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            
            # Print a clean error message
            print(f"\n❌ Training failed: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            
            # Check for CUDA-specific errors and provide debugging info
            error_msg = str(e)
            if "CUDA" in error_msg and torch.cuda.is_available():
                print(f"\n🔍 CUDA Error Debugging Information:")
                print(f"   - CUDA Device: {torch.cuda.get_device_name()}")
                print(f"   - Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                print(f"   - Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
                print(f"   - Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
                
                print(f"\n💡 Debugging suggestions:")
                print(f"   - Try reducing batch_size further (current: {self.config.data.batch_size})")
                print(f"   - Enable gradient checkpointing: use_gradient_checkpointing: true")
                print(f"   - Run with CUDA_LAUNCH_BLOCKING=1 for better error tracing")
                print(f"   - Check GPU driver compatibility")
                print(f"   - Try restarting the training to clear GPU state")
            
            # Save a minimal traceback to logs only
            import traceback
            logger.debug("Full traceback:", exc_info=True)
            
            raise SystemExit(1)
    
    def _train_loop(self):
        """Internal training loop implementation."""
        set_seed(self.config.random_seed)

        # Calculate training parameters based on dataset size
        print("  - Calculating training parameters from dataset...")
        self._calculate_training_parameters()

        # Initialize components with GPU accelerations
        try:
            # Try to use Flash Attention 2 first
            self.model = create_model(
                self.config,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16
            ).to(self.device)
            print("  - Successfully initialized model with Flash Attention 2")
        except (ImportError, ValueError) as e:
            # Fall back to standard attention if Flash Attention is not available
            print(f"  - Flash Attention 2 not available ({e}), falling back to standard attention")
            self.model = create_model(
                self.config,
                torch_dtype=torch.bfloat16
            ).to(self.device)

        # Apply torch.compile for JIT optimization
        print("  - Compiling model with torch.compile()...")
        self.model = torch.compile(self.model, mode="reduce-overhead")
        
        # Apply memory optimizations
        if self.config.training.use_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("  - TF32 enabled for faster training on Ampere+ GPUs")
        
        if self.config.training.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("  - Gradient checkpointing enabled to save memory")
        
        # Initialize memory tracking
        max_memory_reserved = 0
        oom_counter = 0
        gradient_overflow_counter = 0
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
            eps=self.config.training.adam_epsilon
        )
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=self.config.training.train_steps
        )

        self._load_checkpoint()

        # If not resuming, load tokenizer from its original path
        if self.tokenizer is None:
            tokenizer_path = os.path.join(self.base_dir, self.config.tokenizer.output_dir)
            self.tokenizer = self._load_sentencepiece_tokenizer(tokenizer_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        self._prepare_data()

        # Initialize W&B logging
        if self.config.logging.use_wandb:
            wandb.init(
                project=self.config.logging.wandb_project,
                name=self.config.experiment_name,
                config=self.config.model_dump(),
                resume="allow",
                id=wandb.util.generate_id()
            )

        # Save environment snapshot
        self._save_environment_snapshot()

        # Handle checkpoint scheduling
        checkpoint_schedule = self._get_checkpoint_schedule()
        progress_bar = tqdm(range(self.config.training.train_steps), initial=self.global_step, desc="Training Steps")

        # Training metrics tracking
        total_tokens_processed = 0
        
        # Calculate steps per epoch for proper epoch tracking
        steps_per_epoch = self.data_processor.get_training_steps_per_epoch()
        
        self.model.train()
        
        # Standard PyTorch training loop - no custom iterator management
        for epoch in range(self.epoch, self.config.training.epochs):
            if self.global_step >= self.config.training.train_steps:
                break
                
            self.epoch = epoch
            epoch_losses = []
            epoch_start_time = None
            
            # Update progress bar description to show current epoch
            progress_bar.set_description(f"Epoch {epoch + 1}/{self.config.training.epochs}")
            
            print(f"\n--- Epoch {epoch + 1}/{self.config.training.epochs} ---")
            
            # Record start time using simple time tracking instead of CUDA events
            epoch_wall_start = time.time()
            
            for batch in self.dataloader:
                if self.global_step >= self.config.training.train_steps:
                    break

                try:
                    inputs = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                    
                    # Forward pass with AMP
                    if self.amp_enabled:
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            outputs = self.model(**inputs)
                            loss = outputs.loss / self.config.training.gradient_accumulation_steps
                        
                        # Backward with gradient scaling
                        self.scaler.scale(loss).backward()
                        
                        # Gradient accumulation boundary
                        if (self.global_step + 1) % self.config.training.gradient_accumulation_steps == 0:
                            # Unscale gradients for gradient clipping
                            self.scaler.unscale_(self.optimizer)
                            
                            # Gradient clipping (if configured)
                            max_grad_norm = getattr(self.config.training, 'max_grad_norm', None)
                            if max_grad_norm:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), 
                                    max_grad_norm
                                )
                            
                            # Optimizer step with overflow checking
                            scale_before = self.scaler.get_scale()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            scale_after = self.scaler.get_scale()
                            
                            # Check for gradient overflow
                            if scale_after < scale_before:
                                gradient_overflow_counter += 1
                                if gradient_overflow_counter % 10 == 0:
                                    print(f"  ⚠️ Gradient overflow detected (count: {gradient_overflow_counter})")
                            
                            self.lr_scheduler.step()
                            self.optimizer.zero_grad(set_to_none=True)  # More memory efficient
                    else:
                        # Non-AMP path
                        outputs = self.model(**inputs)
                        loss = outputs.loss / self.config.training.gradient_accumulation_steps
                        loss.backward()
                        
                        if (self.global_step + 1) % self.config.training.gradient_accumulation_steps == 0:
                            max_grad_norm = getattr(self.config.training, 'max_grad_norm', None)
                            if max_grad_norm:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), 
                                    max_grad_norm
                                )
                            self.optimizer.step()
                            self.lr_scheduler.step()
                            self.optimizer.zero_grad(set_to_none=True)

                    # Track metrics
                    epoch_losses.append(loss.item() * self.config.training.gradient_accumulation_steps)
                    total_tokens_processed += inputs['input_ids'].numel()
                    
                    # Memory monitoring (every 100 steps)
                    if self.global_step % 100 == 0 and torch.cuda.is_available():
                        current_reserved = torch.cuda.memory_reserved() / 1024**3
                        max_memory_reserved = max(max_memory_reserved, current_reserved)
                        
                        # Only clear cache if memory fragmentation is severe
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        fragmentation = current_reserved - allocated
                        
                        if fragmentation > 4.0:  # More than 4GB fragmented
                            print(f"  ⚠️ High memory fragmentation detected: {fragmentation:.2f}GB")
                            torch.cuda.empty_cache()
                    
                    # Calculate and log metrics
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
                    # Calculate current epoch based on steps completed
                    current_epoch = (self.global_step // steps_per_epoch) + 1
                    
                    # Calculate ETA
                    steps_remaining = self.config.training.train_steps - self.global_step
                    if steps_remaining > 0:
                        time_per_step = progress_bar.format_dict.get('elapsed', 0) / max(1, self.global_step)
                        eta_seconds = steps_remaining * time_per_step
                        eta_str = f"{eta_seconds/3600:.1f}h" if eta_seconds > 3600 else f"{eta_seconds/60:.1f}m"
                    else:
                        eta_str = "0m"
                    
                    # Update progress bar with detailed metrics
                    progress_bar.set_postfix({
                        'loss': f"{avg_loss:.4f}",
                        'lr': f"{current_lr:.2e}",
                        'epoch': f"{current_epoch}/{self.config.training.epochs}",
                        'eta': eta_str
                    })

                    if self.config.logging.use_wandb:
                        wandb.log({
                            "loss": loss.item(), 
                            "learning_rate": current_lr,
                            "epoch": current_epoch,
                            "tokens_processed": total_tokens_processed,
                            "memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
                            "memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
                        }, step=self.global_step)
                    
                    # Checkpoint saving with memory cleanup
                    if self.global_step in checkpoint_schedule:
                        # Ensure all gradients are cleared before checkpoint
                        self.optimizer.zero_grad(set_to_none=True)
                        
                        # Wait for all CUDA operations to complete
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        self._save_checkpoint()
                        
                        # Clear cache after checkpoint to free memory
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    self.global_step += 1
                    progress_bar.update(1)
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        oom_counter += 1
                        print(f"\n⚠️ OOM error #{oom_counter} at step {self.global_step}")
                        print(f"  Memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
                        print(f"  Memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
                        
                        # Clear cache and retry with smaller batch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        
                        # Skip this batch and continue
                        self.optimizer.zero_grad(set_to_none=True)
                        continue
                    else:
                        # Re-raise non-OOM errors
                        raise
            
            # Log epoch completion
            epoch_wall_end = time.time()
            epoch_time = epoch_wall_end - epoch_wall_start
            
            epoch_avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            print(f"  Epoch {epoch + 1} completed:")
            print(f"    - Average loss: {epoch_avg_loss:.4f}")
            print(f"    - Learning rate: {current_lr:.2e}")
            print(f"    - Time: {epoch_time:.1f}s")
            print(f"    - Tokens processed: {total_tokens_processed:,}")
            
            # End of epoch cleanup - minimal intervention
            if torch.cuda.is_available():
                # Only synchronize, don't clear cache unless needed
                torch.cuda.synchronize()
                
                # Log memory stats
                print(f"    - Memory stats - Allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB, "
                      f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f}GB, "
                      f"Max Reserved: {max_memory_reserved:.2f}GB")
                
                # Report gradient overflow stats if any occurred
                if gradient_overflow_counter > 0:
                    print(f"    - Gradient overflows this epoch: {gradient_overflow_counter}")

        print("\n----- Training Complete -----")
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            # Only one final empty_cache
            torch.cuda.empty_cache()
        
        if self.config.logging.use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Run the main training loop for an experiment.")
    parser.add_argument("config_path", type=str, help="Path to the experiment's .yaml config file.")
    args = parser.parse_args()

    base_dir = find_project_root(__file__)
    abs_config_path = args.config_path if os.path.isabs(args.config_path) else os.path.join(base_dir, args.config_path)

    with open(abs_config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    try:
        config = ExperimentConfig(**config_data)
    except Exception as e:
        print(f"FATAL: Error validating configuration file '{abs_config_path}':\n{e}")
        return

    trainer = Trainer(config, base_dir)
    trainer.train()


if __name__ == '__main__':
    main()