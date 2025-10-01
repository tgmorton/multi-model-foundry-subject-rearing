"""
Checkpoint management for model training.

This module handles saving and loading of model checkpoints including
model weights, optimizer state, scheduler state, and training metadata.
"""

import glob
import re
import random
import datetime
import hashlib
import json
from pathlib import Path
from typing import Optional, Set, Dict, Any
import numpy as np
import torch
import wandb


class CheckpointManager:
    """
    Manages model checkpoints during training.

    Handles checkpoint saving/loading, schedule management, and metadata tracking.
    """

    def __init__(self, config, base_dir: str, git_commit_hash: str):
        """
        Initialize the checkpoint manager.

        Args:
            config: Experiment configuration
            base_dir: Project base directory
            git_commit_hash: Current git commit hash
        """
        self.config = config
        self.base_dir = base_dir
        self.git_commit_hash = git_commit_hash
        self.output_dir = Path(base_dir) / config.training.output_dir

    def get_checkpoint_schedule(self) -> Set[int]:
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

    def save_checkpoint(self, model, tokenizer, optimizer, lr_scheduler,
                        global_step: int, epoch: int, scaler: Optional[torch.cuda.amp.GradScaler] = None):
        """
        Save the complete training state to a checkpoint directory.

        Args:
            model: The model to save
            tokenizer: The tokenizer to save
            optimizer: The optimizer state
            lr_scheduler: The learning rate scheduler state
            global_step: Current global training step
            epoch: Current epoch number
            scaler: Optional AMP gradient scaler
        """
        checkpoint_dir = self.output_dir / f"checkpoint-{global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

        # Save training state
        state = {
            'global_step': global_step,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_random_state': torch.get_rng_state(),
            'torch_cuda_random_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'git_commit_hash': self.git_commit_hash,
            # Add AMP scaler state
            'amp_scaler': scaler.state_dict() if scaler is not None else None,
        }
        torch.save(state, checkpoint_dir / "training_state.pt")

        # Save checkpoint metadata
        metadata = {
            'experiment_name': self.config.experiment_name,
            'global_step': global_step,
            'epoch': epoch,
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

        print(f"\n  - Saved checkpoint at step {global_step} to '{checkpoint_dir}'")
        print(f"    - Config hash: {metadata['config_hash'][:8]}...")
        if metadata['wandb_run_id']:
            print(f"    - WandB run ID: {metadata['wandb_run_id']}")

    def load_checkpoint(self, model_factory, device: torch.device,
                        optimizer, lr_scheduler, scaler: Optional[torch.cuda.amp.GradScaler] = None):
        """
        Load training state from the latest checkpoint if resume is enabled.

        Args:
            model_factory: Function to create a new model instance
            device: Device to load the model onto
            optimizer: Optimizer to restore state into
            lr_scheduler: Scheduler to restore state into
            scaler: Optional AMP gradient scaler to restore

        Returns:
            Tuple of (model, tokenizer, global_step, epoch) or (None, None, 0, 0) if no checkpoint
        """
        if not self.config.training.resume_from_checkpoint or not self.output_dir.exists():
            return None, None, 0, 0

        checkpoints = glob.glob(str(self.output_dir / "checkpoint-*"))
        if not checkpoints:
            print("  - `resume_from_checkpoint` is true, but no checkpoints found. Starting fresh.")
            return None, None, 0, 0

        # Find the checkpoint with the highest step number
        latest_checkpoint = max(checkpoints, key=lambda p: int(re.search(r'checkpoint-(\d+)', p).group(1)))
        print(f"  - Resuming training from latest checkpoint: {latest_checkpoint}")

        # Load tokenizer first as it's needed for model setup
        from .tokenization import load_tokenizer
        tokenizer = load_tokenizer(latest_checkpoint)

        # Load model and move to device
        model = model_factory().to(device)
        model.load_state_dict(torch.load(Path(latest_checkpoint) / "pytorch_model.bin", map_location=device))

        # Load training state
        state = torch.load(Path(latest_checkpoint) / "training_state.pt", map_location="cpu")
        global_step = state['global_step']
        epoch = state['epoch']

        # Restore optimizer and scheduler states
        optimizer.load_state_dict(state['optimizer'])
        lr_scheduler.load_state_dict(state['lr_scheduler'])

        # Restore RNG states
        random.setstate(state['random_state'])
        np.random.set_state(state['numpy_random_state'])
        torch.set_rng_state(state['torch_random_state'])
        if torch.cuda.is_available() and state['torch_cuda_random_state']:
            torch.cuda.set_rng_state_all(state['torch_cuda_random_state'])

        # Restore AMP scaler state
        if scaler is not None and state.get('amp_scaler') is not None:
            scaler.load_state_dict(state['amp_scaler'])
            print(f"  - Restored AMP scaler state")

        print(f"  - Resumed from step {global_step} at epoch {epoch}.")

        return model, tokenizer, global_step, epoch

    def cleanup_old_checkpoints(self, keep_latest: int = 5):
        """
        Remove old checkpoints to save disk space.

        Args:
            keep_latest: Number of most recent checkpoints to keep
        """
        if not self.output_dir.exists():
            return

        checkpoints = glob.glob(str(self.output_dir / "checkpoint-*"))
        if len(checkpoints) <= keep_latest:
            return

        # Sort by step number
        checkpoints.sort(key=lambda p: int(re.search(r'checkpoint-(\d+)', p).group(1)))

        # Remove old checkpoints
        for checkpoint in checkpoints[:-keep_latest]:
            import shutil
            shutil.rmtree(checkpoint)
            print(f"  - Removed old checkpoint: {checkpoint}")
