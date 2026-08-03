"""
Core training loop logic.

This module contains the main training loop implementation with support for
mixed precision training, gradient accumulation, and progress tracking.
"""

import os
import time
import logging
from typing import Callable, Optional, Set
import torch
import wandb
from tqdm.auto import tqdm

from .. import registry as _registry
from ..rng import derive_seed


# Type alias for the end-of-epoch hook the sweep agent registers to
# evaluate held-out perplexity, update the running minimum, and signal
# early stopping. Signature:
#   (epoch: int, total_tokens_processed: int) -> bool
# Returning True terminates training after the current epoch.
EpochEndCallback = Callable[[int, int], bool]


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
        # AMP overflow observability (replicability audit): count of
        # optimizer steps the GradScaler skipped due to inf/nan grads.
        self.amp_skipped_steps = 0

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

            # Resume: apply the AMP scaler state the checkpoint manager
            # stashed during load_checkpoint. load_checkpoint runs BEFORE
            # this constructor (in Trainer._train_loop) so it couldn't
            # restore the scaler directly — it had no scaler to restore
            # into. Applying it here keeps the loss-scale continuous across
            # a resume, preventing a spurious overflow/backoff cycle on the
            # first post-resume optimizer step.
            pending = getattr(
                self.checkpoint_manager, "pending_amp_scaler_state", None
            )
            if pending is not None:
                self.scaler.load_state_dict(pending)
                self.checkpoint_manager.pending_amp_scaler_state = None
                self.logger.info("Restored AMP scaler state from checkpoint")

    def run(self, tokenizer, start_step: int = 0, start_epoch: int = 0,
            on_epoch_end: Optional[EpochEndCallback] = None):
        """
        Execute the main training loop.

        Args:
            tokenizer: Tokenizer instance (needed for checkpoint saving)
            start_step: Starting global step (for resuming)
            start_epoch: Starting epoch (for resuming)
            on_epoch_end: Optional hook fired at the end of every epoch
                with (epoch, total_tokens_processed). Returning True stops
                training after the current epoch. Used by the sweep agent
                for per-epoch held-out perplexity + early stopping.

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
        #
        # Precedence:
        #   1. resume_state_steps (explicit set from the schedule helper) —
        #      intersected with the schedule so a non-scheduled step never
        #      persists resume state.
        #   2. save_resume_state_last_n is None → every checkpoint full.
        #   3. otherwise the legacy "last N scheduled steps" suffix.
        last_n = self.config.training.save_resume_state_last_n
        explicit_resume_steps = self.config.training.resume_state_steps
        resume_state_steps: Optional[Set[int]]
        if explicit_resume_steps is not None:
            resume_state_steps = set(explicit_resume_steps) & set(checkpoint_schedule)
        elif last_n is None:
            resume_state_steps = None
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

        # Endpoint guard bookkeeping (STEP 4). Track the last step we wrote
        # a checkpoint at this run so the post-loop guard can avoid a
        # redundant write when training happened to terminate exactly on a
        # scheduled anchor.
        last_saved_step: Optional[int] = None

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
            # Iterator-position counter for the resume fast-forward
            # (replicability C1). Unlike micro_step it is NEVER rewound —
            # the OOM handler rewinds micro_step to the last accumulation
            # boundary, but the batches the OOM'd window pulled from
            # data_iter are gone either way. Persisting THIS counter as
            # epoch_batch_offset keeps "offset == iterator position"
            # exact even across OOM recoveries (implementation-audit m1).
            consumed_from_iter = 0

            # 1.1 — per-optimizer-step timing accumulators. data_ms is wall
            # time spent fetching batches from the dataloader; compute_ms
            # covers H2D transfer + forward + backward + optimizer.step().
            # Reset at each optimizer-step boundary and emitted via
            # _log_metrics so HP sweeps can see the data/compute split.
            data_ms_acc = 0.0
            compute_ms_acc = 0.0
            tokens_acc = 0

            # Replicability redesign: data order is a pure function of
            # (seed, epoch) — reseed the dedicated DataLoader generator and
            # stamp the epoch into the collator BEFORE iter(). Workers are
            # (re)spawned at iter() time and pickle the collator's state,
            # so both must be set first.
            if getattr(self.dataloader, "generator", None) is not None:
                self.dataloader.generator.manual_seed(
                    derive_seed(int(self.config.random_seed), "data_order", epoch)
                )
            _collate = getattr(self.dataloader, "collate_fn", None)
            if hasattr(_collate, "set_epoch"):
                _collate.set_epoch(epoch)

            # Explicit iterator so we can time each next() call cleanly
            # (1.1). Equivalent to the prior `for batch in self.dataloader`.
            data_iter = iter(self.dataloader)

            # Replicability C1 — resume fast-forward. A checkpoint saved
            # mid-epoch records its within-epoch micro-batch position; on
            # the FIRST resumed epoch, skip exactly that many micro-batches
            # so the run continues from where it left off instead of
            # re-training the epoch prefix and truncating the tail. The
            # permutation is identical (per-(seed, epoch) generator above),
            # so skipping lands on the same batches the interrupted run
            # would have seen next. micro_step resumes from its own saved
            # value (epoch_micro_step) — it can lag the iterator position
            # when an OOM rewound an accumulation window before the save.
            resume_offset = int(
                getattr(self.checkpoint_manager, "resume_batch_offset", 0) or 0
            )
            if resume_offset and epoch == start_epoch and start_step > 0:
                self.logger.info(
                    "Resume fast-forward: skipping %d already-consumed "
                    "micro-batches of epoch %d", resume_offset, epoch + 1
                )
                skipped = 0
                for _ in range(resume_offset):
                    try:
                        next(data_iter)
                    except StopIteration:
                        break
                    skipped += 1
                consumed_from_iter = skipped
                micro_step = int(
                    getattr(self.checkpoint_manager, "resume_micro_step", skipped)
                    or skipped
                ) if skipped == resume_offset else skipped
                if skipped != resume_offset:
                    self.logger.warning(
                        "Fast-forward exhausted the epoch after %d/%d "
                        "micro-batches; resuming at next epoch boundary",
                        skipped, resume_offset,
                    )

            while True:
                if self.global_step >= self.config.training.train_steps:
                    break

                t_data_start = time.perf_counter()
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                consumed_from_iter += 1
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
                                current_epoch=self.epoch + 1,
                                train_steps=self.config.training.train_steps,
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
                                # Within-epoch positions for resume
                                # fast-forward (replicability C1):
                                # iterator position (never rewound) and
                                # accumulation counter (OOM-rewindable).
                                epoch_batch_offset=consumed_from_iter,
                                epoch_micro_step=micro_step,
                            )
                            last_saved_step = self.global_step

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

            # End-of-epoch hook for the sweep agent: evaluate held-out
            # perplexity, update BO's running-min metric, check early-stop
            # patience. Returns True to terminate training early.
            if on_epoch_end is not None:
                try:
                    should_stop = bool(on_epoch_end(epoch, total_tokens_processed))
                except Exception as cb_err:  # noqa: BLE001
                    self.logger.warning("on_epoch_end callback raised %s; continuing", cb_err)
                    should_stop = False
                if should_stop:
                    self.logger.info("Early stop requested by epoch callback at epoch %d", epoch)
                    break

        # STEP 4 — endpoint guard. The final reached step is structurally
        # unreachable by any computed schedule anchor: the dataloader
        # iterates PHYSICAL batches (data.py) while train_steps is derived
        # from the EFFECTIVE batch, and the in-loop save-check fires before
        # the global_step increment. So whatever step training actually
        # terminated on may never have been checkpointed. Guarantee an
        # endpoint for BOTH fresh and resumed runs — with full resume state
        # — unless we already saved at exactly this step this run.
        # _save_checkpoint/save_checkpoint use exist_ok, so a redundant call
        # is harmless and idempotent.
        if last_saved_step != self.global_step:
            epoch_completed = (
                steps_per_epoch > 0
                and self.global_step > 0
                and self.global_step == (self.epoch + 1) * steps_per_epoch
            )
            self.logger.info(
                "Endpoint guard: saving final checkpoint at step %d "
                "(last scheduled save was at %s)",
                self.global_step, last_saved_step,
            )
            self._save_checkpoint(
                tokenizer,
                total_tokens_processed,
                save_resume_state=True,
                epoch_completed=epoch_completed,
            )

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
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # Observability (replicability audit): a shrinking scale
                # means inf/nan grads were found — the optimizer step was
                # SKIPPED while the LR scheduler below still advances. A
                # divergent skip decision converts low-bit kernel noise
                # into a hard trajectory fork, so make every skip visible
                # in logs/WandB instead of silent.
                if self.scaler.get_scale() < scale_before:
                    self.amp_skipped_steps += 1
                    self.logger.warning(
                        "AMP overflow at global_step %d: optimizer step "
                        "skipped (scale %.0f -> %.0f; %d skip(s) total)",
                        self.global_step, scale_before,
                        self.scaler.get_scale(), self.amp_skipped_steps,
                    )

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

        # AMP observability (replicability audit): the loss-scale trajectory
        # and skip count make GradScaler-induced trajectory forks visible.
        if self.amp_enabled and self.scaler is not None:
            payload["amp/loss_scale"] = self.scaler.get_scale()
            payload["amp/skipped_steps"] = self.amp_skipped_steps

        wandb.log(payload, step=self.global_step)

    def _save_checkpoint(self, tokenizer, total_tokens_processed: int = 0,
                         save_resume_state: bool = True,
                         epoch_batch_offset: int = 0,
                         epoch_micro_step: int = 0,
                         epoch_completed: bool = False):
        """
        Save a checkpoint with proper cleanup.

        Args:
            tokenizer: Tokenizer to save with checkpoint
            total_tokens_processed: Total number of tokens processed so far
            save_resume_state: If True, include optimizer/scheduler/RNG/AMP
                state so the run can be resumed from this checkpoint. If
                False, write analysis-only (model + tokenizer + metadata),
                which is ~3× smaller on disk. See 0.6.
            epoch_batch_offset: Within-epoch micro-batch position at save
                time, persisted into training_state.pt so resume can
                fast-forward to it (replicability C1). The endpoint guard
                passes 0 — training is complete there.
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
            epoch_batch_offset=epoch_batch_offset,
            epoch_micro_step=epoch_micro_step,
            epoch_completed=epoch_completed,
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
