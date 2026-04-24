"""
Core training loop logic.

This module contains the main training loop implementation with support for
mixed precision training, gradient accumulation, and progress tracking.
"""

import os
import time
import logging
from typing import Optional, Set
import torch
import wandb
from tqdm.auto import tqdm

from .. import registry as _registry


# How often to emit a registry heartbeat. A reaper marks runs PREEMPTED
# after ~2h of no heartbeat, so 5 min gives plenty of margin and is
# cheap (one ~2KB S3 PutObject per step block).
_HEARTBEAT_INTERVAL_S = 300.0


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
            self.scaler = torch.amp.GradScaler('cuda',
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

        # 0.6 — determine which scheduled steps save full resume state
        # (model + optimizer + scheduler + RNG + AMP scaler). Earlier
        # scheduled steps save analysis-only (model + tokenizer + metadata),
        # which is ~3× smaller. ``None`` keeps the legacy behavior where
        # every checkpoint carries full resume state.
        last_n = self.config.training.save_resume_state_last_n
        if last_n is None:
            resume_state_steps: Optional[Set[int]] = None
        else:
            sorted_schedule = sorted(checkpoint_schedule)
            resume_state_steps = (
                set(sorted_schedule[-last_n:]) if last_n > 0 else set()
            )

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

        # Registry heartbeat plumbing — resolved once; reused per step.
        # Mirrors cli.py::_registry_identity so the pod and the CLI agree
        # on which record to touch. Non-fatal everywhere.
        registry_identity = self._registry_identity()
        last_heartbeat_wall = time.time()

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

            # micro_step counts every micro-batch (DataLoader iteration)
            # and drives the gradient accumulation boundary check.
            # global_step only increments on optimizer steps so that
            # train_steps, checkpoint schedules, and logging are all
            # calibrated in optimizer-step units.
            micro_step = 0

            # 1.1 — per-optimizer-step timing accumulators. data_ms is wall
            # time spent fetching batches from the dataloader; compute_ms
            # covers H2D transfer + forward + backward + optimizer.step().
            # Reset at each optimizer-step boundary and emitted via
            # _log_metrics so HP sweeps can see the data/compute split.
            data_ms_acc = 0.0
            compute_ms_acc = 0.0
            tokens_acc = 0

            # Explicit iterator so we can time each next() call cleanly
            # (1.1). Equivalent to the prior `for batch in self.dataloader`.
            data_iter = iter(self.dataloader)

            while True:
                if self.global_step >= self.config.training.train_steps:
                    break

                t_data_start = time.perf_counter()
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                data_ms_acc += (time.perf_counter() - t_data_start) * 1000

                try:
                    t_compute_start = time.perf_counter()

                    # Move batch to device
                    inputs = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

                    # Forward and backward pass. Returns the loss as a
                    # detached GPU tensor (0.3) — do NOT call .item() here
                    # or we pay a cudaDeviceSynchronize on every micro-batch.
                    loss, did_optimizer_step = self._training_step(inputs, micro_step)

                    compute_ms_acc += (time.perf_counter() - t_compute_start) * 1000

                    epoch_losses.append(loss)
                    tokens_this_batch = inputs['input_ids'].numel()
                    total_tokens_processed += tokens_this_batch
                    tokens_acc += tokens_this_batch
                    micro_step += 1

                    # The remaining bookkeeping only runs when the
                    # optimizer actually stepped (accumulation boundary).
                    if did_optimizer_step:
                        # Memory monitoring
                        if self.global_step % 100 == 0 and torch.cuda.is_available():
                            max_memory_reserved = self._monitor_memory(max_memory_reserved)

                        # Registry heartbeat. Gated on wall time rather
                        # than step count so cadence is stable across
                        # arches with very different step durations.
                        # Reads the latest loss tensor without forcing a
                        # sync (loss is already on device; .item() at the
                        # heartbeat boundary is cheap vs. per-micro-batch).
                        now_wall = time.time()
                        if registry_identity is not None and now_wall - last_heartbeat_wall >= _HEARTBEAT_INTERVAL_S:
                            try:
                                loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
                            except Exception:
                                loss_val = None
                            _registry.heartbeat(
                                **registry_identity,
                                current_step=self.global_step,
                                current_loss=loss_val,
                            )
                            last_heartbeat_wall = now_wall

                        # Update progress bar
                        self._update_progress(
                            progress_bar, epoch_losses, steps_per_epoch
                        )

                        # Logging
                        if self.config.logging.use_wandb:
                            self._log_metrics(
                                loss,
                                total_tokens_processed,
                                steps_per_epoch,
                                data_ms=data_ms_acc,
                                compute_ms=compute_ms_acc,
                                tokens_this_step=tokens_acc,
                            )

                        # Checkpoint saving. 0.6 — only the last N scheduled
                        # steps carry optimizer+RNG+scaler state; earlier
                        # saves are analysis-only.
                        if self.global_step in checkpoint_schedule:
                            save_resume = (
                                resume_state_steps is None
                                or self.global_step in resume_state_steps
                            )
                            self._save_checkpoint(
                                tokenizer,
                                total_tokens_processed,
                                save_resume_state=save_resume,
                            )

                        # Reset per-step timing after logging/saving so
                        # checkpoint save time doesn't pollute the next
                        # step's numbers.
                        data_ms_acc = 0.0
                        compute_ms_acc = 0.0
                        tokens_acc = 0

                        self.global_step += 1
                        progress_bar.update(1)

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        oom_counter += 1
                        self._handle_oom(oom_counter, micro_step)
                        # Reset micro_step to the last accumulation
                        # boundary so the window restarts cleanly.
                        grad_accum = self.config.training.gradient_accumulation_steps
                        micro_step = micro_step - (micro_step % grad_accum)
                        # Drop the partial-window timing; next optimizer
                        # step should start from a clean slate.
                        data_ms_acc = 0.0
                        compute_ms_acc = 0.0
                        tokens_acc = 0
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

        # NOTE: do NOT call wandb.finish() here. In sweep contexts, the
        # agent (scripts/sweep_agent_lm.py) needs the run OPEN after
        # train() returns so it can stamp proxy/final_training_loss and
        # proxy/held_out_perplexity on the same trial. wandb's SDK
        # finalises the run automatically on process exit for standalone
        # training runs, so this call is redundant there too.

        return self.global_step

    def _training_step(self, inputs: dict, micro_step: int) -> tuple:
        """
        Execute a single training step (forward + backward).

        Args:
            inputs: Batch of inputs
            micro_step: Current micro-batch index within the gradient
                accumulation window (0-based). The optimizer steps when
                ``(micro_step + 1) % gradient_accumulation_steps == 0``.

        Returns:
            Tuple of (loss_value, did_optimizer_step) where did_optimizer_step
            is True when the optimizer stepped at an accumulation boundary.
        """
        grad_accum = self.config.training.gradient_accumulation_steps
        did_step = False

        if self.amp_enabled:
            # AMP training path
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = self.model(**inputs)
                loss = outputs.loss / grad_accum

            # Backward with gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient accumulation boundary
            if (micro_step + 1) % grad_accum == 0:
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
                did_step = True
        else:
            # Standard training path
            outputs = self.model(**inputs)
            loss = outputs.loss / grad_accum
            loss.backward()

            if (micro_step + 1) % grad_accum == 0:
                max_grad_norm = getattr(self.config.training, 'max_grad_norm', None)
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_grad_norm
                    )
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                did_step = True

        # 0.3 — return the loss as a detached GPU tensor rather than a
        # Python float. Calling .item() here forces cudaDeviceSynchronize
        # on every micro-batch; at gradient_accumulation_steps=16 that's
        # 16 syncs per optimizer step. Callers sync at their own cadence.
        return (loss.detach() * grad_accum), did_step

    def _registry_identity(self):
        """Build the dict of (arch, lang, condition, seed, run_id) that
        keys a registry record for this run. Mirrors
        ``cli.py::_registry_identity`` — launchers set REGISTRY_* env
        vars, smoke/hand-run configs fall back to "unknown" so the
        record still goes somewhere sensible. Returns ``None`` if the
        config is missing the architecture field (shouldn't happen)."""
        try:
            arch = self.config.model.architecture
        except Exception:
            return None
        return {
            "arch": arch,
            "lang": os.environ.get("REGISTRY_LANG")
                    or getattr(self.config, "lang", None)
                    or "unknown",
            "condition": os.environ.get("REGISTRY_CONDITION")
                         or getattr(self.config, "condition", None)
                         or "unknown",
            "run_id": os.environ.get("REGISTRY_RUN_ID")
                      or getattr(self.config, "run_id", None)
                      or self.config.experiment_name,
        }

    @staticmethod
    def _mean_loss(losses: list) -> float:
        """
        Reduce a list of per-micro-batch losses to a scalar float.

        Losses may be either Python floats (legacy) or 0-D tensors (0.3).
        Tensors are stacked and reduced in a single kernel so there is only
        one host/device sync, not one per element.
        """
        if not losses:
            return 0.0
        if isinstance(losses[0], torch.Tensor):
            return torch.stack(losses).mean().item()
        return sum(losses) / len(losses)

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
            epoch_losses: List of per-micro-batch loss tensors for current
                epoch (detached, on-device). We sync to Python scalar here,
                which is one of the few places per optimizer step that a
                host/device sync is acceptable.
            steps_per_epoch: Number of steps per epoch
        """
        current_lr = self.lr_scheduler.get_last_lr()[0]
        avg_loss = self._mean_loss(epoch_losses)
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

    def _log_metrics(self, loss, total_tokens_processed: int, steps_per_epoch: int,
                     data_ms: float = 0.0, compute_ms: float = 0.0,
                     tokens_this_step: int = 0):
        """
        Log metrics to W&B.

        Args:
            loss: Current loss (scalar float or 0-D tensor). Tensor inputs
                are converted via .item() here — acceptable since logging
                is gated on optimizer-step cadence, not micro-batch.
            total_tokens_processed: Total tokens processed so far
            steps_per_epoch: Number of steps per epoch
            data_ms: Time spent fetching this optimizer step's batches from
                the dataloader, in milliseconds (1.1).
            compute_ms: Time spent on H2D + forward + backward + optimizer
                for this optimizer step, in milliseconds (1.1).
            tokens_this_step: Tokens processed in this optimizer step; used
                to derive tokens/sec.
        """
        current_lr = self.lr_scheduler.get_last_lr()[0]
        current_epoch = min(self.config.training.epochs, (self.global_step // steps_per_epoch) + 1)

        if isinstance(loss, torch.Tensor):
            loss_value = loss.item()
        else:
            loss_value = float(loss)

        total_ms = data_ms + compute_ms
        tokens_per_sec = (tokens_this_step / (total_ms / 1000.0)) if total_ms > 0 else 0.0

        payload = {
            "loss": loss_value,
            "learning_rate": current_lr,
            "epoch": current_epoch,
            "tokens_processed": total_tokens_processed,
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
            "memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0,
            # 1.1 — data-vs-compute timing split. Together with
            # tokens_per_sec these distinguish "compute-bound" from
            # "data-bound" runs, which drives HP sweep and CPU-budget
            # tuning decisions.
            "timing/data_ms": data_ms,
            "timing/compute_ms": compute_ms,
            "timing/total_ms": total_ms,
            "timing/data_fraction": (data_ms / total_ms) if total_ms > 0 else 0.0,
            "throughput/tokens_per_sec": tokens_per_sec,
        }

        wandb.log(payload, step=self.global_step)

    def _save_checkpoint(self, tokenizer, total_tokens_processed: int = 0,
                         save_resume_state: bool = True):
        """
        Save a checkpoint with proper cleanup.

        Args:
            tokenizer: Tokenizer to save with checkpoint
            total_tokens_processed: Total number of tokens processed so far
            save_resume_state: If True, include optimizer/scheduler/RNG/AMP
                state so the run can be resumed from this checkpoint. If
                False, write analysis-only (model + tokenizer + metadata),
                which is ~3× smaller on disk. See 0.6.
        """
        # Ensure all gradients are cleared before checkpoint
        self.optimizer.zero_grad(set_to_none=True)

        # Wait for all CUDA operations to complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.checkpoint_manager.save_checkpoint(
            self.model, tokenizer, self.optimizer, self.lr_scheduler,
            self.global_step, self.epoch, self.scaler, total_tokens_processed,
            save_resume_state=save_resume_state,
        )

        # Clear cache after checkpoint to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _handle_oom(self, oom_counter: int, micro_step: int = 0):
        """
        Handle out-of-memory errors.

        Args:
            oom_counter: Number of OOM errors encountered
            micro_step: Current micro-batch index within the accumulation
                window. Used to report how many accumulated micro-batches
                were discarded.
        """
        print(f"\n⚠️ OOM error #{oom_counter} at step {self.global_step}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        print(f"  Memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f}GB")

        # Clear cache and retry
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Skip this batch and discard any accumulated gradients
        self.optimizer.zero_grad(set_to_none=True)

        # Warn if gradient accumulation state was lost
        grad_accum = self.config.training.gradient_accumulation_steps
        if grad_accum > 1:
            lost = micro_step % grad_accum
            if lost != 0:
                print(f"  Warning: Lost {lost} accumulated micro-batches "
                      f"in current gradient accumulation window")

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
        epoch_avg_loss = self._mean_loss(epoch_losses)
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
