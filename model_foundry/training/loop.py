"""
Core training loop logic.

This module contains the main training loop implementation with support for
mixed precision training, gradient accumulation, and progress tracking.
"""

import time
import logging
from typing import Optional, Set
import torch
import wandb
from tqdm.auto import tqdm


class TrainingLoop:
    """
    Manages the core training loop execution.

    Handles forward/backward passes, optimizer steps, logging, and progress tracking.
    """

    def __init__(self, config, model, optimizer, lr_scheduler, dataloader,
                 device: torch.device, checkpoint_manager, data_processor):
        """
        Initialize the training loop.

        Args:
            config: Experiment configuration
            model: The model to train
            optimizer: Optimizer instance
            lr_scheduler: Learning rate scheduler
            dataloader: DataLoader for training data
            device: Device to train on (CPU/CUDA)
            checkpoint_manager: CheckpointManager instance
            data_processor: DataProcessor instance for dataset info
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.dataloader = dataloader
        self.device = device
        self.checkpoint_manager = checkpoint_manager
        self.data_processor = data_processor
        self.logger = logging.getLogger("trainer")

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Setup AMP
        self.scaler = None
        self.amp_enabled = False
        if self.config.training.use_amp and torch.cuda.is_available():
            self.amp_enabled = True
            self.scaler = torch.cuda.amp.GradScaler(
                init_scale=2**16,
                growth_factor=2,
                backoff_factor=0.5,
                growth_interval=2000,
                enabled=True
            )
            self.logger.info("AMP enabled with enhanced GradScaler settings")

    def run(self, tokenizer, start_step: int = 0, start_epoch: int = 0):
        """
        Execute the main training loop.

        Args:
            tokenizer: Tokenizer instance (needed for checkpoint saving)
            start_step: Starting global step (for resuming)
            start_epoch: Starting epoch (for resuming)

        Returns:
            Final global step reached
        """
        self.global_step = start_step
        self.epoch = start_epoch

        # Get checkpoint schedule
        checkpoint_schedule = self.checkpoint_manager.get_checkpoint_schedule()
        progress_bar = tqdm(
            range(self.config.training.train_steps),
            initial=self.global_step,
            desc="Training Steps"
        )

        # Training metrics tracking
        total_tokens_processed = 0
        steps_per_epoch = self.data_processor.get_training_steps_per_epoch()

        # Memory and error tracking
        max_memory_reserved = 0
        oom_counter = 0
        gradient_overflow_counter = 0

        self.model.train()

        self.logger.info("Starting training loop...")
        self.logger.info(f"Epochs to run: {self.epoch} to {self.config.training.epochs}")
        self.logger.info(f"Current global_step: {self.global_step}, target: {self.config.training.train_steps}")

        # Main training loop
        for epoch in range(self.epoch, self.config.training.epochs):
            if self.global_step >= self.config.training.train_steps:
                break

            self.epoch = epoch
            epoch_losses = []
            progress_bar.set_description(f"Epoch {epoch + 1}/{self.config.training.epochs}")

            print(f"\n--- Epoch {epoch + 1}/{self.config.training.epochs} ---")
            epoch_wall_start = time.time()

            for batch_idx, batch in enumerate(self.dataloader):
                if self.global_step >= self.config.training.train_steps:
                    break

                try:
                    # Move batch to device
                    inputs = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

                    # Forward and backward pass
                    loss = self._training_step(inputs)
                    epoch_losses.append(loss)
                    total_tokens_processed += inputs['input_ids'].numel()

                    # Memory monitoring
                    if self.global_step % 100 == 0 and torch.cuda.is_available():
                        max_memory_reserved = self._monitor_memory(max_memory_reserved)

                    # Update progress bar
                    self._update_progress(
                        progress_bar, epoch_losses, steps_per_epoch
                    )

                    # Logging
                    if self.config.logging.use_wandb:
                        self._log_metrics(loss, total_tokens_processed, steps_per_epoch)

                    # Checkpoint saving
                    if self.global_step in checkpoint_schedule:
                        self._save_checkpoint(tokenizer)

                    self.global_step += 1
                    progress_bar.update(1)

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        oom_counter += 1
                        self._handle_oom(oom_counter)
                        continue
                    else:
                        raise

            # Epoch completion
            epoch_wall_end = time.time()
            self._log_epoch_completion(
                epoch, epoch_losses, epoch_wall_end - epoch_wall_start,
                total_tokens_processed, max_memory_reserved, gradient_overflow_counter
            )

            # Epoch cleanup
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        print("\n----- Training Complete -----")

        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if self.config.logging.use_wandb:
            wandb.finish()

        return self.global_step

    def _training_step(self, inputs: dict) -> float:
        """
        Execute a single training step (forward + backward).

        Args:
            inputs: Batch of inputs

        Returns:
            Loss value for this step
        """
        if self.amp_enabled:
            # AMP training path
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
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
        else:
            # Standard training path
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

        return loss.item() * self.config.training.gradient_accumulation_steps

    def _monitor_memory(self, max_memory_reserved: float) -> float:
        """
        Monitor CUDA memory usage and clear cache if needed.

        Args:
            max_memory_reserved: Previous maximum memory reserved

        Returns:
            Updated maximum memory reserved
        """
        current_reserved = torch.cuda.memory_reserved() / 1024**3
        max_memory_reserved = max(max_memory_reserved, current_reserved)

        # Only clear cache if memory fragmentation is severe
        allocated = torch.cuda.memory_allocated() / 1024**3
        fragmentation = current_reserved - allocated

        if fragmentation > 4.0:  # More than 4GB fragmented
            print(f"  ⚠️ High memory fragmentation detected: {fragmentation:.2f}GB")
            torch.cuda.empty_cache()

        return max_memory_reserved

    def _update_progress(self, progress_bar, epoch_losses: list, steps_per_epoch: int):
        """
        Update the progress bar with current metrics.

        Args:
            progress_bar: tqdm progress bar instance
            epoch_losses: List of losses for current epoch
            steps_per_epoch: Number of steps per epoch
        """
        current_lr = self.lr_scheduler.get_last_lr()[0]
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        current_epoch = min(self.config.training.epochs, (self.global_step // steps_per_epoch) + 1)

        # Calculate ETA
        steps_remaining = self.config.training.train_steps - self.global_step
        if steps_remaining > 0:
            time_per_step = progress_bar.format_dict.get('elapsed', 0) / max(1, self.global_step)
            eta_seconds = steps_remaining * time_per_step
            eta_str = f"{eta_seconds/3600:.1f}h" if eta_seconds > 3600 else f"{eta_seconds/60:.1f}m"
        else:
            eta_str = "0m"

        progress_bar.set_postfix({
            'loss': f"{avg_loss:.4f}",
            'lr': f"{current_lr:.2e}",
            'epoch': f"{current_epoch}/{self.config.training.epochs}",
            'eta': eta_str
        })

    def _log_metrics(self, loss: float, total_tokens_processed: int, steps_per_epoch: int):
        """
        Log metrics to W&B.

        Args:
            loss: Current loss value
            total_tokens_processed: Total tokens processed so far
            steps_per_epoch: Number of steps per epoch
        """
        current_lr = self.lr_scheduler.get_last_lr()[0]
        current_epoch = min(self.config.training.epochs, (self.global_step // steps_per_epoch) + 1)

        wandb.log({
            "loss": loss,
            "learning_rate": current_lr,
            "epoch": current_epoch,
            "tokens_processed": total_tokens_processed,
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
            "memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
        }, step=self.global_step)

    def _save_checkpoint(self, tokenizer):
        """
        Save a checkpoint with proper cleanup.

        Args:
            tokenizer: Tokenizer to save with checkpoint
        """
        # Ensure all gradients are cleared before checkpoint
        self.optimizer.zero_grad(set_to_none=True)

        # Wait for all CUDA operations to complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.checkpoint_manager.save_checkpoint(
            self.model, tokenizer, self.optimizer, self.lr_scheduler,
            self.global_step, self.epoch, self.scaler
        )

        # Clear cache after checkpoint to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _handle_oom(self, oom_counter: int):
        """
        Handle out-of-memory errors.

        Args:
            oom_counter: Number of OOM errors encountered
        """
        print(f"\n⚠️ OOM error #{oom_counter} at step {self.global_step}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        print(f"  Memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f}GB")

        # Clear cache and retry
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Skip this batch
        self.optimizer.zero_grad(set_to_none=True)

    def _log_epoch_completion(self, epoch: int, epoch_losses: list, epoch_time: float,
                              total_tokens_processed: int, max_memory_reserved: float,
                              gradient_overflow_counter: int):
        """
        Log epoch completion statistics.

        Args:
            epoch: Epoch number
            epoch_losses: List of losses for the epoch
            epoch_time: Time taken for the epoch
            total_tokens_processed: Total tokens processed
            max_memory_reserved: Maximum memory reserved
            gradient_overflow_counter: Number of gradient overflows
        """
        epoch_avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        current_lr = self.lr_scheduler.get_last_lr()[0]

        print(f"  Epoch {epoch + 1} completed:")
        print(f"    - Average loss: {epoch_avg_loss:.4f}")
        print(f"    - Learning rate: {current_lr:.2e}")
        print(f"    - Time: {epoch_time:.1f}s")
        print(f"    - Tokens processed: {total_tokens_processed:,}")

        if torch.cuda.is_available():
            print(f"    - Memory stats - Allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB, "
                  f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f}GB, "
                  f"Max Reserved: {max_memory_reserved:.2f}GB")

            if gradient_overflow_counter > 0:
                print(f"    - Gradient overflows this epoch: {gradient_overflow_counter}")
