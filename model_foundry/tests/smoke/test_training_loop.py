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
