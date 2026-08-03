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


def resolve_resume_epoch(global_step: int, saved_epoch: int,
                         steps_per_epoch: int, resume_batch_offset: int,
                         epoch_completed: bool) -> int:
    """Resolve the zero-based epoch at which training should resume.

    New endpoint checkpoints explicitly mark a completed epoch. Legacy
    endpoints are recognized only when their zero offset and saved step
    exactly identify the end of the recorded epoch.
    """
    if epoch_completed:
        return saved_epoch + 1
    if (
        global_step > 0
        and steps_per_epoch > 0
        and resume_batch_offset == 0
        and global_step == (saved_epoch + 1) * steps_per_epoch
    ):
        return saved_epoch + 1
    return saved_epoch


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
        # When resuming, load_checkpoint stashes the saved AMP GradScaler
        # state here if no live scaler was passed in (the trainer's
        # GradScaler is built inside TrainingLoop AFTER load_checkpoint
        # runs). TrainingLoop applies it post-construction. None when not
        # resuming or when the checkpoint had no scaler state.
        self.pending_amp_scaler_state: Optional[Dict[str, Any]] = None
        # Replicability C1: within-epoch micro-batch position recovered
        # from training_state.pt by load_checkpoint. The loop fast-forwards
        # the first resumed epoch by this many micro-batches. 0 when not
        # resuming or for checkpoints predating the field.
        self.resume_batch_offset: int = 0
        # Accumulation-counter position paired with resume_batch_offset
        # (differs only when an OOM rewound a window before the save).
        self.resume_micro_step: int = 0
        self.resume_epoch_completed: bool = False

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
                        global_step: int, epoch: int, scaler: Optional[torch.cuda.amp.GradScaler] = None,
                        total_tokens_processed: int = 0,
                        save_resume_state: bool = True,
                        epoch_batch_offset: int = 0,
                        epoch_micro_step: int = 0,
                        epoch_completed: bool = False):
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
            total_tokens_processed: Total number of tokens processed so far
            save_resume_state: If True (default), also write
                ``training_state.pt`` containing optimizer + scheduler +
                RNG + AMP scaler state so the run can be resumed from this
                checkpoint. If False, write analysis-only (model weights,
                tokenizer, metadata). Analysis-only checkpoints are ~3×
                smaller, suitable for older scheduled checkpoints where
                we're unlikely to need resume. See 0.6 in the optimization
                plan.
        """
        checkpoint_dir = self.output_dir / f"checkpoint-{global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

        # Save training state (only when requested — resume state is ~2×
        # the model size and is useless once the run finishes).
        if save_resume_state:
            state = {
                'global_step': global_step,
                'epoch': epoch,
                # Replicability C1: within-epoch positions, so resume can
                # fast-forward the (per-(seed, epoch) seeded) dataloader
                # instead of re-entering the epoch at batch 0.
                # epoch_batch_offset = micro-batches CONSUMED from the
                # iterator (never rewound); epoch_micro_step = the
                # accumulation counter, which can lag after an OOM rewind.
                'epoch_batch_offset': epoch_batch_offset,
                'epoch_micro_step': epoch_micro_step,
                'epoch_completed': epoch_completed,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'random_state': random.getstate(),
                'numpy_random_state': np.random.get_state(),
                'torch_random_state': torch.get_rng_state(),
                'torch_cuda_random_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'git_commit_hash': self.git_commit_hash,
                # Add AMP scaler state
                'amp_scaler': scaler.state_dict() if scaler is not None else None,
                # Add token count
                'total_tokens_processed': total_tokens_processed,
            }
            torch.save(state, checkpoint_dir / "training_state.pt")

        # Calculate token metrics for comparison
        batch_size = self.config.data.batch_size
        seq_length = self.config.data.max_sequence_length
        grad_accum = self.config.training.gradient_accumulation_steps
        effective_batch_size = batch_size * grad_accum
        tokens_per_step = effective_batch_size * seq_length

        # Save checkpoint metadata
        metadata = {
            'experiment_name': self.config.experiment_name,
            'global_step': global_step,
            'epoch': epoch,
            'timestamp': datetime.datetime.now().isoformat(),
            'git_commit_hash': self.git_commit_hash,
            'config_hash': hashlib.md5(json.dumps(self.config.model_dump(), sort_keys=True).encode()).hexdigest(),
            'wandb_run_id': wandb.run.id if self.config.logging.use_wandb and wandb.run else None,
            # 0.6 — record whether this checkpoint can be resumed from
            # (full state) or is analysis-only. Used by loaders and the
            # post-eval pruner.
            'has_resume_state': save_resume_state,
            'epoch_completed': epoch_completed,
            # Token counting metrics for cross-architecture comparison
            'token_metrics': {
                'total_tokens_processed': total_tokens_processed,
                'tokens_per_step': tokens_per_step,
                'effective_batch_size': effective_batch_size,
                'sequence_length': seq_length,
                'estimated_tokens_at_step': global_step * tokens_per_step,
            },
            'training_config': {
                'learning_rate': self.config.training.learning_rate,
                'batch_size': self.config.data.batch_size,
                'gradient_accumulation_steps': self.config.training.gradient_accumulation_steps,
                'use_amp': self.config.training.use_amp,
                'use_tf32': self.config.training.use_tf32,
                'use_gradient_checkpointing': self.config.training.use_gradient_checkpointing,
            },
            'model_config': {
                'architecture': self.config.model.architecture,
                'layers': self.config.model.layers,
                'hidden_size': self.config.model.hidden_size,
                'attention_heads': self.config.model.attention_heads,
            }
        }

        with open(checkpoint_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        kind = "full resume" if save_resume_state else "analysis-only"
        print(f"\n  - Saved checkpoint at step {global_step} to '{checkpoint_dir}' ({kind})")
        print(f"    - Tokens processed: {total_tokens_processed:,} ({total_tokens_processed/1e6:.2f}M)")
        print(f"    - Config hash: {metadata['config_hash'][:8]}...")
        if metadata['wandb_run_id']:
            print(f"    - WandB run ID: {metadata['wandb_run_id']}")

    def load_checkpoint(self, model, device: torch.device,
                        optimizer, lr_scheduler, scaler: Optional[torch.cuda.amp.GradScaler] = None):
        """
        Load training state from the latest checkpoint if resume is enabled.

        Loads model weights INTO the provided model object in-place (rather
        than creating a new one), which is critical: the optimizer is bound
        to ``model.parameters()`` at construction time, so replacing the
        model object would silently break optimizer.step() (gradients would
        accumulate on the loaded model but the optimizer would iterate the
        old/discarded parameter refs, recording no inf checks and tripping
        ``scaler.step()`` with "No inf checks were recorded for this optimizer").

        Args:
            model: Existing model object to load weights into (in-place).
            device: Device the model is on (used for safetensors loader).
            optimizer: Optimizer to restore state into
            lr_scheduler: Scheduler to restore state into
            scaler: Optional AMP gradient scaler to restore

        Returns:
            Tuple of (tokenizer, global_step, epoch) or (None, 0, 0) if no checkpoint
        """
        if not self.config.training.resume_from_checkpoint or not self.output_dir.exists():
            return None, 0, 0

        checkpoints = glob.glob(str(self.output_dir / "checkpoint-*"))
        if not checkpoints:
            print("  - `resume_from_checkpoint` is true, but no checkpoints found. Starting fresh.")
            return None, 0, 0

        # 0.6 — only analysis-only checkpoints (no training_state.pt) may
        # exist from earlier scheduled saves. We need the latest checkpoint
        # that actually carries a resume state; falling back to an analysis
        # checkpoint would silently lose optimizer/scheduler/RNG state.
        def _step_of(path: str) -> int:
            return int(re.search(r'checkpoint-(\d+)', path).group(1))

        checkpoints_sorted = sorted(checkpoints, key=_step_of, reverse=True)
        latest_checkpoint = None
        for candidate in checkpoints_sorted:
            if (Path(candidate) / "training_state.pt").exists():
                latest_checkpoint = candidate
                break

        if latest_checkpoint is None:
            print("  - `resume_from_checkpoint` is true, but no checkpoint carries "
                  "a training_state.pt (all are analysis-only). Starting fresh.")
            return None, 0, 0

        if latest_checkpoint != checkpoints_sorted[0]:
            skipped = _step_of(checkpoints_sorted[0]) - _step_of(latest_checkpoint)
            print(f"  - Skipped {skipped} analysis-only checkpoint step(s) with no "
                  f"resume state; resuming from {latest_checkpoint}.")
        else:
            print(f"  - Resuming training from latest checkpoint: {latest_checkpoint}")

        # Load tokenizer first as it's needed for model setup
        from .tokenization import load_tokenizer
        tokenizer = load_tokenizer(latest_checkpoint)

        # Load weights into the existing model in-place.
        # HF save_pretrained may emit either model.safetensors (default in
        # newer transformers) or pytorch_model.bin. The state dict keys
        # correspond to the underlying HF model (e.g. "transformer.h.0..."),
        # not our wrapper class which prefixes with "hf_model.". Load into
        # model.hf_model when present, and use strict=False to allow tied
        # weights (e.g. GPT-2's lm_head) to be missing from the saved state.
        ckpt_path = Path(latest_checkpoint)
        safetensors_file = ckpt_path / "model.safetensors"
        bin_file = ckpt_path / "pytorch_model.bin"
        if safetensors_file.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(safetensors_file), device=str(device))
        elif bin_file.exists():
            state_dict = torch.load(bin_file, map_location=device,
                                    weights_only=False)
        else:
            raise FileNotFoundError(
                f"No model weights found in {latest_checkpoint}. "
                f"Expected model.safetensors or pytorch_model.bin."
            )

        target = model.hf_model if hasattr(model, "hf_model") else model
        missing, unexpected = target.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys loading checkpoint: {unexpected}"
            )
        # Tied weights are legitimately missing from save_pretrained output
        # (they share storage with another param and are reconstructed via
        # tie_weights). Consult the model's own declaration of which keys
        # are tied, falling back to a permissive name-based check for
        # implementations that don't populate _tied_weights_keys.
        allowed_missing = set()
        for attr_name in ("_tied_weights_keys",):
            attr = getattr(target, attr_name, None)
            if attr:
                allowed_missing.update(attr)
        name_patterns = (
            "lm_head",                        # GPT-2
            "cls.predictions.decoder",         # BERT MLM head
            "tied",
        )
        for k in missing:
            if k in allowed_missing:
                continue
            if any(pat in k for pat in name_patterns):
                continue
            raise RuntimeError(
                f"Unexpected missing key loading checkpoint: {k}"
            )

        # Load training state. mmap=True keeps the (potentially multi-GB)
        # optimizer tensors backed by the file rather than copied wholesale
        # into a peak RAM spike — load_state_dict below pulls only what it
        # needs. weights_only=False is required because the state dict holds
        # arbitrary Python objects (RNG tuples, numpy state) beyond tensors.
        state = torch.load(
            Path(latest_checkpoint) / "training_state.pt",
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        global_step = state['global_step']
        epoch = state['epoch']
        # Replicability C1: stash the within-epoch positions for the loop's
        # resume fast-forward. Back-compatible: checkpoints written before
        # these fields default to 0 (re-enter the epoch at batch 0 — the old
        # behavior). resume_micro_step defaults to the batch offset (they
        # only differ when an OOM rewound an accumulation window pre-save).
        self.resume_batch_offset = int(state.get('epoch_batch_offset', 0) or 0)
        self.resume_micro_step = int(
            state.get('epoch_micro_step', self.resume_batch_offset)
            or self.resume_batch_offset
        )
        self.resume_epoch_completed = bool(state.get('epoch_completed', False))

        # Restore optimizer and scheduler states
        optimizer.load_state_dict(state['optimizer'])
        lr_scheduler.load_state_dict(state['lr_scheduler'])

        # Restore RNG states
        random.setstate(state['random_state'])
        np.random.set_state(state['numpy_random_state'])
        torch.set_rng_state(state['torch_random_state'])
        if torch.cuda.is_available() and state['torch_cuda_random_state']:
            torch.cuda.set_rng_state_all(state['torch_cuda_random_state'])

        # Restore AMP scaler state. The trainer calls load_checkpoint
        # BEFORE the GradScaler exists (it's constructed in
        # TrainingLoop.__init__), so ``scaler`` is typically None here. In
        # that case, stash the saved state on the manager so TrainingLoop
        # can apply it to the freshly-built scaler. If a live scaler IS
        # passed (e.g. a future caller that builds it earlier, or tests),
        # restore directly.
        saved_scaler_state = state.get('amp_scaler')
        if saved_scaler_state is not None:
            if scaler is not None:
                scaler.load_state_dict(saved_scaler_state)
                print(f"  - Restored AMP scaler state")
            else:
                self.pending_amp_scaler_state = saved_scaler_state
                print(f"  - Stashed AMP scaler state for post-construction restore")

        print(f"  - Resumed from step {global_step} at epoch {epoch}.")

        return tokenizer, global_step, epoch

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
