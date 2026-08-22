"""
Smoke tests for model_foundry.training.loop.TrainingLoop.

All tests run on CPU without real data. Heavy dependencies (model, optimizer,
scheduler, checkpoint manager, data processor) are replaced with mocks.
"""

import types
from unittest.mock import MagicMock, patch, call

import pytest
import torch

from model_foundry.training.loop import TrainingLoop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_model(loss_value: float = 1.0):
    """Return a mock model whose forward pass yields a differentiable loss."""
    model = MagicMock()
    # parameters() must return real Parameters so zero_grad / clip_grad work
    param = torch.nn.Parameter(torch.randn(4, 4))
    model.parameters.return_value = [param]

    def forward(**kwargs):
        # Build a real scalar tensor that supports .backward()
        loss = param.sum() * 0 + loss_value  # graph-attached constant
        out = types.SimpleNamespace(loss=loss)
        return out

    model.side_effect = forward
    model.__call__ = forward
    model.train = MagicMock()
    return model


def _make_config(
    gradient_accumulation_steps: int = 1,
    use_amp: bool = False,
    train_steps: int = 10,
    epochs: int = 1,
    use_wandb: bool = False,
    max_grad_norm: float = None,
):
    """Build a minimal config namespace matching the attributes TrainingLoop reads."""
    training = types.SimpleNamespace(
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_amp=use_amp,
        train_steps=train_steps,
        epochs=epochs,
        warmup_steps=2,
        max_grad_norm=max_grad_norm,
    )
    logging_cfg = types.SimpleNamespace(use_wandb=use_wandb)
    config = types.SimpleNamespace(training=training, logging=logging_cfg)
    return config


def _make_loop(config=None, model=None, **overrides):
    """Construct a TrainingLoop with mocked-out heavy dependencies."""
    if config is None:
        config = _make_config(**{k: v for k, v in overrides.items()
                                  if k in ('gradient_accumulation_steps',
                                           'use_amp', 'train_steps',
                                           'epochs', 'use_wandb',
                                           'max_grad_norm')})
    if model is None:
        model = _make_mock_model(overrides.get('loss_value', 1.0))

    optimizer = MagicMock()
    lr_scheduler = MagicMock()
    lr_scheduler.get_last_lr.return_value = [1e-4]
    checkpoint_manager = MagicMock()
    checkpoint_manager.get_checkpoint_schedule.return_value = set()
    data_processor = MagicMock()
    data_processor.get_training_steps_per_epoch.return_value = 100

    device = torch.device('cpu')

    loop = TrainingLoop(
        config=config,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=MagicMock(),   # not used directly in these tests
        device=device,
        checkpoint_manager=checkpoint_manager,
        data_processor=data_processor,
    )
    return loop


def _dummy_batch(seq_len: int = 8):
    """Return a minimal {input_ids, attention_mask, labels} batch on CPU."""
    ids = torch.randint(0, 100, (2, seq_len))
    return {
        'input_ids': ids,
        'attention_mask': torch.ones_like(ids),
        'labels': ids.clone(),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrainingStep:
    """_training_step returns (loss_value, did_optimizer_step)."""

    def test_no_optimizer_step_before_boundary(self):
        """When micro_step+1 is NOT a multiple of grad_accum, no optimizer step."""
        loop = _make_loop(gradient_accumulation_steps=4)
        batch = _dummy_batch()

        loss_val, did_step = loop._training_step(batch, micro_step=0)

        assert isinstance(loss_val, float)
        assert did_step is False
        loop.optimizer.step.assert_not_called()
        loop.lr_scheduler.step.assert_not_called()

    def test_optimizer_step_at_boundary(self):
        """When micro_step+1 IS a multiple of grad_accum, optimizer steps."""
        loop = _make_loop(gradient_accumulation_steps=4)
        batch = _dummy_batch()

        loss_val, did_step = loop._training_step(batch, micro_step=3)

        assert isinstance(loss_val, float)
        assert did_step is True
        loop.optimizer.step.assert_called_once()
        loop.lr_scheduler.step.assert_called_once()
        loop.optimizer.zero_grad.assert_called_once_with(set_to_none=True)

    def test_single_accumulation_always_steps(self):
        """With gradient_accumulation_steps=1, every micro_step triggers a step."""
        loop = _make_loop(gradient_accumulation_steps=1)
        batch = _dummy_batch()

        _, did_step = loop._training_step(batch, micro_step=0)
        assert did_step is True

    def test_loss_value_reflects_model_output(self):
        """Returned loss should equal the model's raw loss (after / * grad_accum round-trip)."""
        loop = _make_loop(gradient_accumulation_steps=2, loss_value=4.0)
        batch = _dummy_batch()

        loss_val, _ = loop._training_step(batch, micro_step=0)
        assert abs(loss_val - 4.0) < 1e-5


class TestGlobalStepIncrement:
    """global_step only increments on optimizer steps, not every micro-batch."""

    def test_global_step_counts_optimizer_steps(self):
        """With grad_accum=4 and 8 micro-batches, global_step should increment 2 times."""
        grad_accum = 4
        n_micro = 8
        loop = _make_loop(
            gradient_accumulation_steps=grad_accum,
            train_steps=100,
            epochs=1,
        )

        # Build a fake dataloader that yields n_micro batches
        batches = [_dummy_batch() for _ in range(n_micro)]

        # Simulate what run() does: iterate micro-batches, increment
        # global_step only when did_optimizer_step is True.
        micro_step = 0
        for batch in batches:
            inputs = {k: v.to(loop.device) for k, v in batch.items()}
            _, did_step = loop._training_step(inputs, micro_step)
            micro_step += 1
            if did_step:
                loop.global_step += 1

        assert loop.global_step == n_micro // grad_accum  # 2

    def test_no_extra_increments(self):
        """global_step must not increment on non-boundary micro-steps."""
        loop = _make_loop(gradient_accumulation_steps=3, train_steps=100)

        micro_step = 0
        for i in range(6):
            _, did_step = loop._training_step(_dummy_batch(), micro_step)
            micro_step += 1
            if did_step:
                loop.global_step += 1

        # 6 micro-batches / 3 accum = 2 optimizer steps
        assert loop.global_step == 2


class TestOOMRecovery:
    """_handle_oom clears gradients and micro_step resets to accumulation boundary."""

    def test_handle_oom_zeros_gradients(self):
        loop = _make_loop(gradient_accumulation_steps=4)

        with patch('torch.cuda.is_available', return_value=False):
            loop._handle_oom(oom_counter=1, micro_step=5)

        loop.optimizer.zero_grad.assert_called_once_with(set_to_none=True)

    def test_micro_step_reset_to_boundary(self):
        """After OOM at micro_step=5 with grad_accum=4,
        micro_step should snap back to 4 (last boundary)."""
        grad_accum = 4
        micro_step = 5

        # Reproduce the reset logic from run()
        micro_step = micro_step - (micro_step % grad_accum)

        assert micro_step == 4  # 5 - (5 % 4) = 5 - 1 = 4

    def test_micro_step_reset_at_boundary(self):
        """If OOM happens exactly at a boundary, micro_step stays the same."""
        grad_accum = 4
        micro_step = 4

        micro_step = micro_step - (micro_step % grad_accum)

        assert micro_step == 4  # already aligned

    def test_micro_step_reset_early_in_window(self):
        """OOM at micro_step=1 with grad_accum=4 resets to 0."""
        grad_accum = 4
        micro_step = 1

        micro_step = micro_step - (micro_step % grad_accum)

        assert micro_step == 0

    def test_handle_oom_warns_about_lost_microbatches(self, capsys):
        """When grad_accum > 1 and micro_step is mid-window, a warning is printed."""
        loop = _make_loop(gradient_accumulation_steps=4)

        with patch('torch.cuda.is_available', return_value=False):
            loop._handle_oom(oom_counter=1, micro_step=2)

        captured = capsys.readouterr()
        assert "Lost 2 accumulated micro-batches" in captured.out


class TestAMPDisabledPath:
    """Training step works correctly with use_amp=False (CPU path)."""

    def test_no_scaler_created(self):
        loop = _make_loop(use_amp=False)
        assert loop.scaler is None
        assert loop.amp_enabled is False

    def test_training_step_cpu(self):
        """Forward + backward completes without AMP on CPU."""
        loop = _make_loop(use_amp=False, gradient_accumulation_steps=1)
        batch = _dummy_batch()

        loss_val, did_step = loop._training_step(batch, micro_step=0)

        assert isinstance(loss_val, float)
        assert did_step is True
        loop.optimizer.step.assert_called_once()

    def test_training_step_cpu_with_grad_clip(self):
        """Gradient clipping fires on the CPU path when max_grad_norm is set."""
        loop = _make_loop(
            use_amp=False,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
        )
        batch = _dummy_batch()

        with patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
            _, did_step = loop._training_step(batch, micro_step=0)

        assert did_step is True
        mock_clip.assert_called_once()


class TestCheckpointSchedule:
    """_save_checkpoint is called only at steps listed in the checkpoint schedule."""

    def test_save_at_scheduled_step(self):
        """Checkpoint is saved when global_step is in the schedule."""
        loop = _make_loop(
            gradient_accumulation_steps=1,
            train_steps=100,
        )
        loop.checkpoint_manager.get_checkpoint_schedule.return_value = {5, 10}
        mock_tokenizer = MagicMock()

        # Simulate reaching step 5
        loop.global_step = 5
        schedule = loop.checkpoint_manager.get_checkpoint_schedule()

        if loop.global_step in schedule:
            with patch('torch.cuda.is_available', return_value=False):
                loop._save_checkpoint(mock_tokenizer, total_tokens_processed=1000)

        loop.checkpoint_manager.save_checkpoint.assert_called_once()

    def test_no_save_outside_schedule(self):
        """Checkpoint is NOT saved when global_step is not in the schedule."""
        loop = _make_loop(
            gradient_accumulation_steps=1,
            train_steps=100,
        )
        loop.checkpoint_manager.get_checkpoint_schedule.return_value = {5, 10}

        loop.global_step = 3
        schedule = loop.checkpoint_manager.get_checkpoint_schedule()

        if loop.global_step in schedule:
            loop._save_checkpoint(MagicMock(), total_tokens_processed=0)

        loop.checkpoint_manager.save_checkpoint.assert_not_called()

    def test_save_checkpoint_calls_correct_args(self):
        """_save_checkpoint passes all required arguments to checkpoint_manager."""
        loop = _make_loop(gradient_accumulation_steps=1, train_steps=100)
        loop.global_step = 7
        loop.epoch = 2
        mock_tokenizer = MagicMock()

        with patch('torch.cuda.is_available', return_value=False):
            loop._save_checkpoint(mock_tokenizer, total_tokens_processed=5000)

        loop.checkpoint_manager.save_checkpoint.assert_called_once_with(
            loop.model,
            mock_tokenizer,
            loop.optimizer,
            loop.lr_scheduler,
            7,   # global_step
            2,   # epoch
            loop.scaler,
            5000,  # total_tokens_processed
            save_resume_state=True,
            epoch_batch_offset=0,
            epoch_micro_step=0,
            epoch_completed=False,
        )


# ---------------------------------------------------------------------------
# Preemption-smoke regression (2026-08-22): save/resume token-counter
# consistency at the TrainingLoop.run() level.
# ---------------------------------------------------------------------------
#
# The checkpoint save inside run()'s did_optimizer_step block fires BEFORE
# `self.global_step += 1` for the window that just completed. So at the
# moment of a mid-loop scheduled save, self.global_step still holds the
# value from BEFORE this window (call it S), while total_tokens_processed
# already includes this window's tokens (S+1 windows' worth) — the
# per-micro-batch accumulation above runs unconditionally, ahead of the
# save. That mismatch is invisible in an uninterrupted run (self.global_step
# catches up two lines later, in the same process) but becomes real once a
# NEW process re-seeds self.global_step from the persisted S: the resumed
# run's own total_tokens_processed permanently carries one extra window's
# worth of tokens relative to an uninterrupted reference run reaching the
# same global_step. The fix (loop.py, the checkpoint-save call site) saves
# `total_tokens_processed - tokens_acc` instead, so the persisted pair
# (global_step, total_tokens_processed) always describes the SAME window
# count.


class _FiniteDataLoader:
    """Fake dataloader yielding a fixed list of micro-batches once.

    Deliberately has no ``.generator``/``.collate_fn`` attributes so
    TrainingLoop.run()'s per-epoch reseed logic (gated on
    ``getattr(..., None) is not None``) stays inert — it's unrelated to the
    token-counter bug under test here.
    """

    def __init__(self, batches):
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)


def _microbatch(n_tokens: int = 12):
    """A single micro-batch with an exact, known token count (1 row)."""
    ids = torch.randint(0, 100, (1, n_tokens))
    return {
        'input_ids': ids,
        'attention_mask': torch.ones_like(ids),
        'labels': ids.clone(),
    }


def _make_run_loop(train_steps, grad_accum, checkpoint_schedule, n_batches):
    """Build a TrainingLoop wired for a full run() call.

    Unlike ``_make_loop`` above (used only for the lower-level
    ``_training_step``/``_save_checkpoint`` unit tests), this wires a real
    finite dataloader and a checkpoint_manager mock that records every
    save_checkpoint call, with just enough config surface for run() to
    execute end-to-end without touching anything outside this module's
    control.
    """
    training = types.SimpleNamespace(
        gradient_accumulation_steps=grad_accum,
        use_amp=False,
        train_steps=train_steps,
        epochs=1,
        warmup_steps=0,
        max_grad_norm=None,
        # 0.6 resume-state precedence fields, read directly by run(). Both
        # None means "every checkpoint saves full state" (legacy default) —
        # irrelevant to token bookkeeping, but run() dereferences them
        # unconditionally so they must exist.
        save_resume_state_last_n=None,
        resume_state_steps=None,
    )
    logging_cfg = types.SimpleNamespace(use_wandb=False)
    config = types.SimpleNamespace(training=training, logging=logging_cfg)

    model = _make_mock_model()
    optimizer = MagicMock()
    lr_scheduler = MagicMock()
    lr_scheduler.get_last_lr.return_value = [1e-4]

    checkpoint_manager = MagicMock()
    checkpoint_manager.get_checkpoint_schedule.return_value = set(checkpoint_schedule)
    # Explicit, not left to MagicMock auto-vivification: getattr(mock,
    # "resume_batch_offset", 0) on a bare MagicMock returns a truthy
    # auto-created Mock attribute, not the intended falsy default, which
    # would wrongly enter the resume fast-forward branch.
    checkpoint_manager.resume_batch_offset = 0
    checkpoint_manager.resume_micro_step = 0

    data_processor = MagicMock()
    data_processor.get_training_steps_per_epoch.return_value = 1_000_000

    dataloader = _FiniteDataLoader([_microbatch() for _ in range(n_batches)])

    return TrainingLoop(
        config=config,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
        device=torch.device('cpu'),
        checkpoint_manager=checkpoint_manager,
        data_processor=data_processor,
    )


class TestTokenCounterSaveResumeConsistency:
    """Regression coverage for the preemption-smoke find (2026-08-22).

    All numbers below are exact and were chosen (grad_accum=2, 12
    tokens/micro-batch => 24 tokens/effective-step) so that the pre-fix
    code fails these assertions by exactly one effective step's worth of
    tokens (24), mirroring the cluster smoke's 128,000-token discrepancy
    at the same 200-of-400 style mid-run schedule anchor.
    """

    GRAD_ACCUM = 2
    TOKENS_PER_MICROBATCH = 12
    TOKENS_PER_STEP = GRAD_ACCUM * TOKENS_PER_MICROBATCH  # 24
    TRAIN_STEPS = 6
    # Mid-run anchor strictly below TRAIN_STEPS — like the smoke's
    # checkpoint_schedule=[200, 400] with train_steps=400, this is never
    # reachable via the natural end-of-loop path (see the endpoint-guard
    # comment in loop.py), only via the mid-loop scheduled-save path whose
    # off-by-one this test targets.
    SCHEDULE = {3}

    def test_mid_loop_checkpoint_records_self_consistent_tokens(self):
        """checkpoint-3's persisted total_tokens_processed must describe 3
        windows' worth of tokens (matching its own global_step label), not
        4 windows' worth (the in-flight window that hasn't yet been
        reflected in self.global_step at save time).
        """
        with patch('torch.cuda.is_available', return_value=False):
            loop = _make_run_loop(
                train_steps=self.TRAIN_STEPS,
                grad_accum=self.GRAD_ACCUM,
                checkpoint_schedule=self.SCHEDULE,
                # Exactly enough micro-batches to complete real step 4 (the
                # window whose completion trips the global_step==3
                # scheduled save), then run dry — mirrors "SIGKILLed
                # shortly after checkpoint-N landed".
                n_batches=4 * self.GRAD_ACCUM,
            )
            loop.run(tokenizer=MagicMock(), start_step=0, start_epoch=0,
                     start_tokens=0)

        calls = loop.checkpoint_manager.save_checkpoint.call_args_list
        assert len(calls) >= 1
        scheduled_call = calls[0]

        assert scheduled_call.args[4] == 3  # global_step label
        assert scheduled_call.args[7] == 3 * self.TOKENS_PER_STEP  # 72, not 96

    def test_resumed_run_matches_uninterrupted_reference_token_total(self):
        """End-to-end: a run split by a simulated preemption at
        checkpoint-3 and then resumed must reach the SAME final
        total_tokens_processed as an uninterrupted reference run — not one
        effective step (TOKENS_PER_STEP) higher.
        """
        # --- Reference: a single, uninterrupted run to train_steps. ---
        with patch('torch.cuda.is_available', return_value=False):
            ref_loop = _make_run_loop(
                train_steps=self.TRAIN_STEPS,
                grad_accum=self.GRAD_ACCUM,
                checkpoint_schedule=self.SCHEDULE,
                n_batches=self.TRAIN_STEPS * self.GRAD_ACCUM,
            )
            ref_loop.run(tokenizer=MagicMock(), start_step=0, start_epoch=0,
                         start_tokens=0)
        ref_calls = ref_loop.checkpoint_manager.save_checkpoint.call_args_list
        reference_final_step = ref_calls[-1].args[4]
        reference_final_tokens = ref_calls[-1].args[7]
        assert reference_final_step == self.TRAIN_STEPS
        assert reference_final_tokens == self.TRAIN_STEPS * self.TOKENS_PER_STEP  # 144

        # --- "Killed shortly after checkpoint-3 landed": only enough data
        # to complete real step 4 (the window whose completion trips the
        # global_step==3 scheduled save), then the process goes dark,
        # exactly like the first run above. ---
        with patch('torch.cuda.is_available', return_value=False):
            killed_loop = _make_run_loop(
                train_steps=self.TRAIN_STEPS,
                grad_accum=self.GRAD_ACCUM,
                checkpoint_schedule=self.SCHEDULE,
                n_batches=4 * self.GRAD_ACCUM,
            )
            killed_loop.run(tokenizer=MagicMock(), start_step=0,
                             start_epoch=0, start_tokens=0)
        killed_calls = killed_loop.checkpoint_manager.save_checkpoint.call_args_list
        loaded_start_step = killed_calls[0].args[4]
        loaded_start_tokens = killed_calls[0].args[7]

        # --- Resume: a fresh process re-seeded from exactly what the
        # checkpoint persisted (mirrors checkpoint_manager.load_checkpoint
        # -> resume_total_tokens -> run(start_tokens=...)). No fast-forward
        # needed — the fake dataloader below IS the remaining data. ---
        with patch('torch.cuda.is_available', return_value=False):
            resumed_loop = _make_run_loop(
                train_steps=self.TRAIN_STEPS,
                grad_accum=self.GRAD_ACCUM,
                checkpoint_schedule=set(),  # avoid re-tripping schedule={3}
                # self.global_step re-enters at loaded_start_step and must
                # climb to TRAIN_STEPS by its OWN increments regardless of
                # the token-counter fix — that structurally requires
                # exactly (TRAIN_STEPS - loaded_start_step) more real
                # optimizer-step windows here, independent of what this
                # test is checking.
                n_batches=(self.TRAIN_STEPS - loaded_start_step) * self.GRAD_ACCUM,
            )
            resumed_loop.run(tokenizer=MagicMock(),
                              start_step=loaded_start_step, start_epoch=0,
                              start_tokens=loaded_start_tokens)
        resumed_calls = resumed_loop.checkpoint_manager.save_checkpoint.call_args_list
        resumed_final_step = resumed_calls[-1].args[4]
        resumed_final_tokens = resumed_calls[-1].args[7]

        assert resumed_final_step == reference_final_step == self.TRAIN_STEPS
        # The regression: pre-fix code leaves this TOKENS_PER_STEP (one
        # whole effective step) higher than the reference.
        assert resumed_final_tokens == reference_final_tokens
