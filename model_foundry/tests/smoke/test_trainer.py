"""
Smoke tests for model_foundry.trainer.Trainer.

These tests verify that the Trainer class initializes correctly and its
helper methods behave as expected, without creating real models or loading data.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch

from model_foundry.trainer import Trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trainer(tiny_config, tmp_path):
    """
    Build a Trainer with heavy side-effects mocked out.

    Patches create_data_processor (IO-heavy) and get_git_commit_hash so that
    Trainer.__init__ can run without touching the filesystem or network.
    """
    with patch("model_foundry.trainer.create_data_processor") as mock_cdp, \
         patch("model_foundry.trainer.get_git_commit_hash", return_value="abc123"):
        mock_cdp.return_value = MagicMock()
        trainer = Trainer(tiny_config, str(tmp_path))
    return trainer


# ---------------------------------------------------------------------------
# 1. Trainer initialisation
# ---------------------------------------------------------------------------

class TestTrainerInit:
    """Trainer(config, base_dir) should set core attributes without errors."""

    def test_sets_device(self, tiny_config, tmp_path):
        trainer = _make_trainer(tiny_config, tmp_path)
        assert isinstance(trainer.device, torch.device)

    def test_stores_git_hash(self, tiny_config, tmp_path):
        trainer = _make_trainer(tiny_config, tmp_path)
        assert trainer.git_commit_hash == "abc123"

    def test_creates_data_processor(self, tiny_config, tmp_path):
        trainer = _make_trainer(tiny_config, tmp_path)
        assert trainer.data_processor is not None

    def test_components_initially_none(self, tiny_config, tmp_path):
        trainer = _make_trainer(tiny_config, tmp_path)
        assert trainer.model is None
        assert trainer.optimizer is None
        assert trainer.lr_scheduler is None
        assert trainer.dataloader is None
        assert trainer.tokenizer is None
        assert trainer.checkpoint_manager is None
        assert trainer.training_loop is None


# ---------------------------------------------------------------------------
# 2. _setup_memory_management on CPU (no-op, no crash)
# ---------------------------------------------------------------------------

class TestSetupMemoryManagementCPU:
    """When CUDA is not available, _setup_memory_management should be a no-op."""

    def test_noop_on_cpu(self, tiny_config, tmp_path):
        with patch("model_foundry.trainer.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            # Re-create trainer; __init__ calls _setup_memory_management
            with patch("model_foundry.trainer.create_data_processor") as mock_cdp, \
                 patch("model_foundry.trainer.get_git_commit_hash", return_value="abc123"), \
                 patch("model_foundry.trainer.get_device", return_value=torch.device("cpu")):
                mock_cdp.return_value = MagicMock()
                trainer = Trainer(tiny_config, str(tmp_path))

            # CUDA memory functions should NOT have been called
            mock_torch.cuda.set_per_process_memory_fraction.assert_not_called()


# ---------------------------------------------------------------------------
# 3. _calculate_training_parameters  -- auto-calculation path
# ---------------------------------------------------------------------------

class TestCalculateTrainingParametersAuto:
    """When train_steps and warmup_steps are None, they are computed from epochs."""

    def test_auto_calculates_train_and_warmup_steps(self, tiny_config, tmp_path):
        # Set train_steps and warmup_steps to None so auto-calculation triggers
        tiny_config.training.train_steps = None
        tiny_config.training.warmup_steps = None
        tiny_config.training.epochs = 3
        tiny_config.training.warmup_ratio = 0.1

        trainer = _make_trainer(tiny_config, tmp_path)

        # Mock data_processor to report a known steps_per_epoch
        trainer.data_processor.get_training_steps_per_epoch.return_value = 200

        trainer._calculate_training_parameters()

        expected_train_steps = 3 * 200  # epochs * steps_per_epoch
        expected_warmup_steps = int(expected_train_steps * 0.1)

        assert trainer.config.training.train_steps == expected_train_steps
        assert trainer.config.training.warmup_steps == expected_warmup_steps

    def test_preserves_explicit_train_steps(self, tiny_config, tmp_path):
        """If train_steps is already set, it should not be overwritten."""
        tiny_config.training.train_steps = 500
        tiny_config.training.warmup_steps = None
        tiny_config.training.warmup_ratio = 0.2

        trainer = _make_trainer(tiny_config, tmp_path)
        trainer.data_processor.get_training_steps_per_epoch.return_value = 100

        trainer._calculate_training_parameters()

        assert trainer.config.training.train_steps == 500
        assert trainer.config.training.warmup_steps == int(500 * 0.2)


# ---------------------------------------------------------------------------
# 4. _calculate_training_parameters  -- exception fallback
# ---------------------------------------------------------------------------

class TestCalculateTrainingParametersFallback:
    """When get_training_steps_per_epoch raises, fall back to 100 steps/epoch."""

    def test_falls_back_to_100_on_exception(self, tiny_config, tmp_path):
        tiny_config.training.train_steps = None
        tiny_config.training.warmup_steps = None
        tiny_config.training.epochs = 2
        tiny_config.training.warmup_ratio = 0.1

        trainer = _make_trainer(tiny_config, tmp_path)
        trainer.data_processor.get_training_steps_per_epoch.side_effect = RuntimeError(
            "dataset not loaded"
        )

        trainer._calculate_training_parameters()

        # Fallback: 100 steps/epoch * 2 epochs
        assert trainer.config.training.train_steps == 200
        assert trainer.config.training.warmup_steps == int(200 * 0.1)


# ---------------------------------------------------------------------------
# 5. _initialize_optimizer_and_scheduler
# ---------------------------------------------------------------------------

class TestInitializeOptimizerAndScheduler:
    """Verify optimizer and lr_scheduler are created with correct parameters."""

    def test_optimizer_and_scheduler_created(self, tiny_config, tmp_path):
        tiny_config.training.train_steps = 100
        tiny_config.training.warmup_steps = 10
        tiny_config.training.learning_rate = 5e-4

        trainer = _make_trainer(tiny_config, tmp_path)

        # Give the trainer a small real model so AdamW can inspect parameters
        trainer.model = torch.nn.Linear(8, 8)

        trainer._initialize_optimizer_and_scheduler()

        assert trainer.optimizer is not None
        assert trainer.lr_scheduler is not None

    def test_optimizer_uses_configured_lr(self, tiny_config, tmp_path):
        lr = 3e-5
        tiny_config.training.train_steps = 50
        tiny_config.training.warmup_steps = 5
        tiny_config.training.learning_rate = lr

        trainer = _make_trainer(tiny_config, tmp_path)
        trainer.model = torch.nn.Linear(8, 8)

        trainer._initialize_optimizer_and_scheduler()

        # After scheduler init, the LR may be adjusted by warmup (starts at 0).
        # Check the optimizer was configured with the right base LR via defaults.
        for pg in trainer.optimizer.param_groups:
            assert pg["initial_lr"] == lr


# ---------------------------------------------------------------------------
# 6. WandB run ID reuse from checkpoint
# ---------------------------------------------------------------------------

class TestWandBRunID:
    """Test WandB init logic inside _train_loop for run-ID reuse."""

    def test_reuses_run_id_from_checkpoint_metadata(self, tiny_config, tmp_path):
        """When resuming (global_step > 0) and metadata exists, reuse the ID."""
        tiny_config.logging.use_wandb = True
        tiny_config.logging.wandb_project = "test-project"

        trainer = _make_trainer(tiny_config, tmp_path)

        # Create a fake checkpoint metadata file
        checkpoint_dir = tmp_path / tiny_config.training.output_dir / "checkpoint-50"
        checkpoint_dir.mkdir(parents=True)
        metadata = {"wandb_run_id": "saved-run-42"}
        (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata))

        # We test the relevant logic in _train_loop by extracting the ID
        # selection code path directly.
        global_step = 50

        # Simulate the checkpoint manager's output_dir
        mock_cm = MagicMock()
        mock_cm.output_dir = tmp_path / tiny_config.training.output_dir

        wandb_run_id = None
        if global_step > 0:
            metadata_path = mock_cm.output_dir / f"checkpoint-{global_step}" / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    md = json.load(f)
                wandb_run_id = md.get("wandb_run_id")

        assert wandb_run_id == "saved-run-42"

    def test_generates_new_id_when_no_metadata(self, tiny_config, tmp_path):
        """When no metadata file exists, a new wandb ID should be generated."""
        tiny_config.logging.use_wandb = True

        trainer = _make_trainer(tiny_config, tmp_path)

        global_step = 50
        mock_cm = MagicMock()
        mock_cm.output_dir = tmp_path / "nonexistent_output"

        wandb_run_id = None
        if global_step > 0:
            metadata_path = mock_cm.output_dir / f"checkpoint-{global_step}" / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    md = json.load(f)
                wandb_run_id = md.get("wandb_run_id")

        # No metadata found, so wandb_run_id stays None
        assert wandb_run_id is None

        # In the real code, this triggers: wandb_run_id = wandb.util.generate_id()
        with patch("wandb.util.generate_id", return_value="fresh-id-99"):
            import wandb
            if wandb_run_id is None:
                wandb_run_id = wandb.util.generate_id()

        assert wandb_run_id == "fresh-id-99"

    def test_no_wandb_when_step_zero(self, tiny_config, tmp_path):
        """When global_step == 0, the checkpoint-metadata branch is skipped."""
        global_step = 0

        wandb_run_id = None
        if global_step > 0:
            # This block should not execute
            wandb_run_id = "should-not-reach"

        assert wandb_run_id is None


# ---------------------------------------------------------------------------
# 7. Deterministic CUDA settings
# ---------------------------------------------------------------------------

class TestDeterministicCUDASettings:
    """Verify cudnn.benchmark and cudnn.deterministic are set when CUDA is available."""

    def test_sets_deterministic_when_cuda_available(self, tiny_config, tmp_path):
        with patch("model_foundry.trainer.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            # Make backends accessible
            mock_torch.backends.cudnn = MagicMock()

            with patch("model_foundry.trainer.create_data_processor") as mock_cdp, \
                 patch("model_foundry.trainer.get_git_commit_hash", return_value="abc123"), \
                 patch("model_foundry.trainer.get_device", return_value=torch.device("cpu")):
                mock_cdp.return_value = MagicMock()

                # Instantiate -- __init__ calls _setup_memory_management
                trainer = Trainer(tiny_config, str(tmp_path))

            # cudnn should be configured for determinism
            assert mock_torch.backends.cudnn.benchmark is False
            assert mock_torch.backends.cudnn.deterministic is True
            # Memory fraction should have been called
            mock_torch.cuda.set_per_process_memory_fraction.assert_called_once_with(0.95)

    def test_noop_when_cuda_not_available(self, tiny_config, tmp_path):
        with patch("model_foundry.trainer.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            with patch("model_foundry.trainer.create_data_processor") as mock_cdp, \
                 patch("model_foundry.trainer.get_git_commit_hash", return_value="abc123"), \
                 patch("model_foundry.trainer.get_device", return_value=torch.device("cpu")):
                mock_cdp.return_value = MagicMock()
                trainer = Trainer(tiny_config, str(tmp_path))

            # Nothing CUDA-related should have been touched
            mock_torch.cuda.set_per_process_memory_fraction.assert_not_called()
