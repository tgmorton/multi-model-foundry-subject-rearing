"""
Smoke tests for model_foundry.training.loop.TrainingLoop.

All tests run on CPU without real data. Heavy dependencies (model, optimizer,
scheduler, checkpoint manager, data processor) are replaced with mocks.
"""

import copy
import types
from unittest.mock import MagicMock, patch, call

import pytest
import torch

from model_foundry.training.checkpointing import resolve_resume_epoch
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
    # Matches _microbatch(): 1 row x 12 tokens. run() reads it to credit the
    # in-flight window's tokens when resuming from a mid-epoch checkpoint.
    data_cfg = types.SimpleNamespace(batch_size=1, max_sequence_length=12)
    config = types.SimpleNamespace(training=training, logging=logging_cfg,
                                   data=data_cfg)

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
        checkpoint-3 and then resumed (through the real fast-forward) must
        reach the SAME final step and total_tokens_processed as an
        uninterrupted reference run.
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
        killed_save = killed_loop.checkpoint_manager.save_checkpoint.call_args_list[0]
        loaded_start_step = killed_save.args[4]
        loaded_start_tokens = killed_save.args[7]

        # --- Resume: a fresh process re-seeded from exactly what the
        # checkpoint persisted (mirrors load_checkpoint -> resume_batch_offset
        # / resume_micro_step / resume_total_tokens -> run()). It gets the
        # FULL dataloader and the same schedule, so the real fast-forward
        # path runs, as on the cluster. ---
        with patch('torch.cuda.is_available', return_value=False):
            resumed_loop = _make_run_loop(
                train_steps=self.TRAIN_STEPS,
                grad_accum=self.GRAD_ACCUM,
                checkpoint_schedule=self.SCHEDULE,
                n_batches=self.TRAIN_STEPS * self.GRAD_ACCUM,
            )
            cm = resumed_loop.checkpoint_manager
            cm.resume_batch_offset = killed_save.kwargs['epoch_batch_offset']
            cm.resume_micro_step = killed_save.kwargs['epoch_micro_step']
            resumed_loop.run(tokenizer=MagicMock(),
                              start_step=loaded_start_step, start_epoch=0,
                              start_tokens=loaded_start_tokens)
        resumed_calls = resumed_loop.checkpoint_manager.save_checkpoint.call_args_list
        resumed_final_step = resumed_calls[-1].args[4]
        resumed_final_tokens = resumed_calls[-1].args[7]

        # The resumed process must not re-save the step it resumed from.
        assert [c.args[4] for c in resumed_calls] == [self.TRAIN_STEPS]
        assert resumed_final_step == reference_final_step == self.TRAIN_STEPS
        # 2026-08-22 regression: the token count must not carry an extra
        # TOKENS_PER_STEP relative to the reference.
        assert resumed_final_tokens == reference_final_tokens


class TestRollingResume:
    """Rolling resume (2026-10-06): extra resume states between the
    permanent anchors, each followed by a prune of the superseded ones."""

    def _run(self, rolling_every):
        with patch('torch.cuda.is_available', return_value=False):
            loop = _make_run_loop(train_steps=12, grad_accum=1,
                                  checkpoint_schedule=set(range(1, 12)),
                                  n_batches=12)
            loop.config.training.resume_state_steps = [2, 11]
            loop.config.training.rolling_resume_every_steps = rolling_every
            loop.run(tokenizer=MagicMock(), start_step=0, start_epoch=0,
                     start_tokens=0)
        saves = {c.args[4]: c.kwargs['save_resume_state']
                 for c in loop.checkpoint_manager.save_checkpoint.call_args_list}
        prunes = [c.args for c in
                  loop.checkpoint_manager.prune_rolling_resume_states.call_args_list]
        return saves, prunes

    def test_rolling_states_between_permanent_anchors(self):
        saves, prunes = self._run(rolling_every=3)
        # 2 and 11 are permanent; 5 and 8 are rolling (>= 3 steps after the
        # previous resume save); 12 is the endpoint guard.
        assert {s for s, full in saves.items() if full} == {2, 5, 8, 11, 12}
        assert [p[0] for p in prunes] == [2, 5, 8, 11, 12]
        assert all(p[1] == {2, 11} for p in prunes)

    def test_off_keeps_permanent_policy(self):
        saves, prunes = self._run(rolling_every=None)
        assert {s for s, full in saves.items() if full} == {2, 11, 12}
        assert prunes == []


class TestStepEndHook:
    """on_step_end (rater sweeps, 2026-10-16): fired once per optimizer
    step with the number of completed steps; returning True stops."""

    def _loop(self, train_steps=6, grad_accum=2, schedule=(), n_batches=None):
        with patch('torch.cuda.is_available', return_value=False):
            return _make_run_loop(
                train_steps=train_steps, grad_accum=grad_accum,
                checkpoint_schedule=set(schedule),
                n_batches=n_batches if n_batches is not None else train_steps * grad_accum,
            )

    def _run(self, loop, **kwargs):
        with patch('torch.cuda.is_available', return_value=False):
            return loop.run(tokenizer=MagicMock(), start_step=0, start_epoch=0,
                            start_tokens=0, **kwargs)

    def test_called_once_per_optimizer_step_with_completed_count(self):
        seen = []
        loop = self._loop(train_steps=6, grad_accum=2)
        final = self._run(loop, on_step_end=lambda s: seen.append(s) or False)
        # Micro-batches that don't close an accumulation window don't fire.
        assert seen == [1, 2, 3, 4, 5, 6]
        assert final == 6

    def test_true_stops_training_and_skips_epoch_hook(self):
        epoch_calls = []
        loop = self._loop(train_steps=10, grad_accum=1)
        final = self._run(
            loop,
            on_step_end=lambda s: s >= 4,
            on_epoch_end=lambda e, t: epoch_calls.append(e) or False,
        )
        assert final == 4
        assert epoch_calls == []
        # The endpoint guard still saves exactly one (full) checkpoint at 4.
        calls = loop.checkpoint_manager.save_checkpoint.call_args_list
        assert [c.args[4] for c in calls] == [4]
        assert calls[0].kwargs['save_resume_state'] is True

    def test_raising_hook_does_not_stop(self):
        def boom(step):
            raise RuntimeError("eval exploded")
        loop = self._loop(train_steps=5, grad_accum=1)
        assert self._run(loop, on_step_end=boom) == 5

    def test_unused_hook_matches_default_run(self):
        """No hook + default endpoint mode == the historical save pattern."""
        a = self._loop(train_steps=6, grad_accum=2, schedule={3})
        b = self._loop(train_steps=6, grad_accum=2, schedule={3})
        self._run(a)
        self._run(b, on_step_end=None, endpoint_checkpoint="full")
        sig = lambda l: [(c.args[4], c.args[7], c.kwargs['save_resume_state'])
                         for c in l.checkpoint_manager.save_checkpoint.call_args_list]
        assert sig(a) == sig(b) == [(3, 72, True), (6, 144, True)]


class TestEndpointCheckpointMode:
    """endpoint_checkpoint controls only the post-loop endpoint guard."""

    def _saves(self, mode, schedule=()):
        with patch('torch.cuda.is_available', return_value=False):
            loop = _make_run_loop(train_steps=4, grad_accum=1,
                                  checkpoint_schedule=set(schedule), n_batches=4)
            loop.run(tokenizer=MagicMock(), start_step=0, start_epoch=0,
                     start_tokens=0, endpoint_checkpoint=mode)
        return [(c.args[4], c.kwargs['save_resume_state'])
                for c in loop.checkpoint_manager.save_checkpoint.call_args_list]

    def test_weights_only_endpoint(self):
        assert self._saves("weights") == [(4, False)]

    def test_no_endpoint(self):
        assert self._saves("none") == []

    def test_scheduled_saves_unaffected(self):
        assert self._saves("none", schedule={2}) == [(2, True)]

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError):
            self._saves("bogus")


# ---------------------------------------------------------------------------
# Resume off-by-one (2026-10-13): save -> kill -> resume through the REAL
# fast-forward path, compared checkpoint by checkpoint with an
# uninterrupted run.
# ---------------------------------------------------------------------------
#
# The in-loop save fires before `self.global_step += 1`, so checkpoint-S
# holds the state after S+1 optimizer windows and records
# epoch_batch_offset = those S+1 windows' micro-batches. Resuming at
# global_step = S made the first post-resume window re-save checkpoint-S one
# window later, and every later label lagged its data by one window per
# resume (rolling-resume smoke 2026-10-06: resumed from 590, skipped 4728 =
# 591*8 micro-batches, then re-saved checkpoint-590). A real tiny model,
# optimizer and LR scheduler make each checkpoint's weights a fingerprint of
# exactly which data it has seen.

_GA = 2                          # gradient accumulation
_B, _L = 2, 4                    # physical batch rows x sequence length
_TPS = _GA * _B * _L             # tokens per optimizer step
_BATCHES_PER_EPOCH = 12
_SPE = _BATCHES_PER_EPOCH // _GA  # 6 optimizer windows per epoch
_EPOCHS = 2
_TRAIN_STEPS = _EPOCHS * _SPE     # 12
_ALL_LABELS = range(_TRAIN_STEPS)  # every in-loop save label


class _Preempted(Exception):
    """Raised by the fake manager right after a save lands (pod killed)."""


class _TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = input_ids.to(torch.float64).mean() / 100
        return types.SimpleNamespace(loss=((self.w - x) ** 2).sum())


class _EpochData:
    """Deterministic micro-batches whose content depends on (epoch, index).
    run() calls collate_fn.set_epoch(epoch) before iter(), as with the real
    collator; there is no .generator, so the reseed branch stays inert."""

    def __init__(self):
        self.collate_fn = self
        self._epoch = 0

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __iter__(self):
        for i in range(_BATCHES_PER_EPOCH):
            ids = torch.full((_B, _L), 1 + 100 * self._epoch + i)
            yield {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
                   'labels': ids.clone()}


class _DiskCheckpointManager:
    """Stands in for CheckpointManager. ``disk`` (label -> record) outlives
    the process, like the PVC; ``kill_after`` kills the process right after
    that label lands."""

    def __init__(self, disk, schedule, kill_after=None):
        self.disk = disk
        self.schedule = set(schedule)
        self.kill_after = kill_after
        self.saved = []  # labels this process wrote, in order
        self.resume_batch_offset = 0
        self.resume_micro_step = 0
        self.pending_amp_scaler_state = None

    def get_checkpoint_schedule(self):
        return set(self.schedule)

    def save_checkpoint(self, model, tokenizer, optimizer, lr_scheduler,
                        global_step, epoch, scaler=None,
                        total_tokens_processed=0, save_resume_state=True,
                        epoch_batch_offset=0, epoch_micro_step=0,
                        epoch_completed=False):
        old = self.disk.get(global_step)
        self.disk[global_step] = {
            'weights': model.w.detach().clone(),
            'opt_steps': lr_scheduler.last_epoch,
            'optimizer': copy.deepcopy(optimizer.state_dict()),
            'lr_scheduler': copy.deepcopy(lr_scheduler.state_dict()),
            'epoch': epoch,
            'total_tokens': total_tokens_processed,
            'epoch_batch_offset': epoch_batch_offset,
            'epoch_micro_step': epoch_micro_step,
            'epoch_completed': epoch_completed,
            # save_checkpoint never downgrades a resumable checkpoint.
            'resume_state': save_resume_state or bool(old and old['resume_state']),
        }
        self.saved.append(global_step)
        if global_step == self.kill_after:
            raise _Preempted(global_step)

    def prune_rolling_resume_states(self, keep_step, permanent_steps):
        pass


def _pod(disk, schedule, kill_after=None, resume_from=None, **training):
    """One pod's lifetime: build model/optimizer/scheduler, resume from
    disk[resume_from] the way load_checkpoint + Trainer do, then run()."""
    model = _TinyLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: 1.0 / (1.0 + 0.1 * s))
    fields = dict(gradient_accumulation_steps=_GA, use_amp=False,
                  train_steps=_TRAIN_STEPS, epochs=_EPOCHS, max_grad_norm=None,
                  save_resume_state_last_n=None, resume_state_steps=None,
                  rolling_resume_every_steps=None)
    fields.update(training)
    config = types.SimpleNamespace(
        training=types.SimpleNamespace(**fields),
        logging=types.SimpleNamespace(use_wandb=False),
        data=types.SimpleNamespace(batch_size=_B, max_sequence_length=_L),
    )
    manager = _DiskCheckpointManager(disk, schedule, kill_after)
    start_step = start_epoch = start_tokens = 0
    if resume_from is not None:
        state = disk[resume_from]
        assert state['resume_state'], resume_from
        model.w.data.copy_(state['weights'])
        optimizer.load_state_dict(state['optimizer'])
        lr_scheduler.load_state_dict(state['lr_scheduler'])
        offset = state['epoch_batch_offset']
        manager.resume_batch_offset = offset
        manager.resume_micro_step = state['epoch_micro_step'] or offset
        start_step, start_tokens = resume_from, state['total_tokens']
        start_epoch = resolve_resume_epoch(
            resume_from, state['epoch'], _SPE, offset, state['epoch_completed'])
        if start_epoch != state['epoch']:
            manager.resume_batch_offset = manager.resume_micro_step = 0
    data_processor = MagicMock()
    data_processor.get_training_steps_per_epoch.return_value = _SPE
    loop = TrainingLoop(config, model, optimizer, lr_scheduler, _EpochData(),
                        torch.device('cpu'), manager, data_processor)
    try:
        with patch('torch.cuda.is_available', return_value=False):
            loop.run(tokenizer=None, start_step=start_step,
                     start_epoch=start_epoch, start_tokens=start_tokens)
    except _Preempted:
        pass
    return manager.saved


def _run_with_preemptions(kill_points, schedule=_ALL_LABELS, **training):
    """Kill the pod right after each label in ``kill_points`` lands and
    resume the next pod from it. Returns (disk, per-pod save logs)."""
    disk, logs, resume_from = {}, [], None
    for kill in list(kill_points) + [None]:
        logs.append(_pod(disk, schedule, kill_after=kill,
                         resume_from=resume_from, **training))
        resume_from = kill
    return disk, logs


def _assert_same_checkpoint(got, ref, label):
    assert got['opt_steps'] == ref['opt_steps'], label
    assert torch.equal(got['weights'], ref['weights']), label
    assert got['total_tokens'] == ref['total_tokens'], label
    assert got['epoch'] == ref['epoch'], label
    assert got['epoch_batch_offset'] == ref['epoch_batch_offset'], label


class TestResumeMatchesUninterrupted:

    def test_reference_label_convention(self):
        """Pins the convention the resume path has to invert: in-loop
        checkpoint-S holds S+1 optimizer steps; the endpoint guard's
        checkpoint-E holds E. Tokens always equal label * tokens_per_step."""
        disk, logs = _run_with_preemptions(())
        assert logs == [list(range(_TRAIN_STEPS + 1))]
        for label, rec in disk.items():
            in_loop = label < _TRAIN_STEPS
            assert rec['opt_steps'] == (label + 1 if in_loop else label)
            assert rec['total_tokens'] == label * _TPS
            assert rec['epoch_completed'] == (not in_loop)

    @pytest.mark.parametrize("kill_points", [
        (3,),        # mid epoch 1
        (0,),        # checkpoint-0 (start_step == 0)
        (5,),        # last window of epoch 1: fast-forward drains the epoch
        (6,),        # first window of epoch 2
        (11,),       # last in-loop save: resumed pod trains nothing
        (3, 8),      # resumes in both epochs
        (2, 3, 4),   # back-to-back resumes (the lag used to accumulate)
    ], ids=lambda k: "kill@" + "+".join(map(str, k)))
    def test_every_checkpoint_matches_reference(self, kill_points):
        ref_disk, ref_logs = _run_with_preemptions(())
        disk, logs = _run_with_preemptions(kill_points)

        # Every label written exactly once, in order: a resumed pod never
        # re-saves the label it resumed from, and the run still ends with
        # the endpoint guard's checkpoint-<train_steps>.
        assert [s for log in logs for s in log] == ref_logs[0]
        assert disk.keys() == ref_disk.keys()
        for label in ref_disk:
            _assert_same_checkpoint(disk[label], ref_disk[label], label)
            assert disk[label]['total_tokens'] == label * _TPS

    def test_rolling_resume_cadence_survives_resume(self):
        rolling = dict(resume_state_steps=[2, 11], rolling_resume_every_steps=3)
        ref_disk, ref_logs = _run_with_preemptions((), **rolling)
        disk, logs = _run_with_preemptions((5, 8), **rolling)

        def full(d):
            return sorted(s for s, r in d.items() if r['resume_state'])

        assert full(ref_disk) == [2, 5, 8, 11, _TRAIN_STEPS]
        assert full(disk) == full(ref_disk)
        assert [s for log in logs for s in log] == ref_logs[0]
        for label in ref_disk:
            _assert_same_checkpoint(disk[label], ref_disk[label], label)

    def test_endpoint_checkpoint_resumes_without_shift(self):
        """The endpoint guard saves AFTER the increment (offset 0), so
        resuming from it must not add a window: a finished run re-saves only
        its identical endpoint."""
        disk, _ = _run_with_preemptions(())
        ref = copy.deepcopy(disk[_TRAIN_STEPS])
        saved = _pod(disk, _ALL_LABELS, resume_from=_TRAIN_STEPS)
        assert saved == [_TRAIN_STEPS]
        got = disk[_TRAIN_STEPS]
        assert got['opt_steps'] == ref['opt_steps'] == _TRAIN_STEPS
        assert torch.equal(got['weights'], ref['weights'])
        assert got['total_tokens'] == ref['total_tokens'] == _TRAIN_STEPS * _TPS

    def test_completed_epoch_endpoint_extends_like_uninterrupted(self):
        """A 1-epoch run's endpoint (epoch_completed) extended to 2 epochs
        matches an uninterrupted 2-epoch run from there on."""
        ref_disk, _ = _run_with_preemptions(())
        disk = {}
        _pod(disk, _ALL_LABELS, epochs=1, train_steps=_SPE)
        assert disk[_SPE]['epoch_completed']
        assert disk[_SPE]['epoch_batch_offset'] == 0
        _pod(disk, _ALL_LABELS, resume_from=_SPE)
        for label in range(_SPE, _TRAIN_STEPS + 1):
            _assert_same_checkpoint(disk[label], ref_disk[label], label)