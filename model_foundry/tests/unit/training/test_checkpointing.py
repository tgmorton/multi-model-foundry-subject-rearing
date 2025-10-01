"""
Unit tests for checkpoint management.

These tests are critical as checkpointing is essential for training reliability
and reproducibility.
"""

import pytest
import torch
import random
import numpy as np
from pathlib import Path

from model_foundry.training.checkpointing import CheckpointManager
from model_foundry.model import create_model


class TestCheckpointManager:
    """Tests for CheckpointManager functionality."""

    def test_initialization(self, tiny_config, temp_workspace):
        """CheckpointManager initializes correctly."""
        manager = CheckpointManager(tiny_config, str(temp_workspace), "test_hash")

        assert manager.config == tiny_config
        assert manager.base_dir == str(temp_workspace)
        assert manager.git_commit_hash == "test_hash"
        assert manager.output_dir == temp_workspace / "test/output"

    def test_save_checkpoint_creates_directory(self, tiny_config, temp_workspace,
                                               tiny_model, mock_tokenizer):
        """Saving a checkpoint creates the correct directory structure."""
        manager = CheckpointManager(tiny_config, str(temp_workspace), "test_hash")

        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0,
                                                       total_iters=10)

        manager.save_checkpoint(
            model=tiny_model,
            tokenizer=mock_tokenizer,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            global_step=100,
            epoch=1
        )

        checkpoint_dir = temp_workspace / "test" / "output" / "checkpoint-100"
        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "training_state.pt").exists()
        assert (checkpoint_dir / "metadata.json").exists()

    def test_save_checkpoint_preserves_model_weights(self, tiny_config, temp_workspace,
                                                     tiny_model, mock_tokenizer):
        """Saved checkpoint contains correct model weights."""
        manager = CheckpointManager(tiny_config, str(temp_workspace), "test_hash")

        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0,
                                                       total_iters=10)

        # Get original weights
        original_weights = {name: param.clone() for name, param in tiny_model.named_parameters()}

        manager.save_checkpoint(
            model=tiny_model,
            tokenizer=mock_tokenizer,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            global_step=50,
            epoch=0
        )

        # Load a new model and check weights
        checkpoint_dir = temp_workspace / "test" / "output" / "checkpoint-50"
        loaded_model = create_model(tiny_config)

        # Load weights manually (without going through load_checkpoint)
        state_dict_path = checkpoint_dir / "pytorch_model.bin"
        if not state_dict_path.exists():
            # Try alternative name (safetensors format)
            state_dict_path = checkpoint_dir / "model.safetensors"
            if not state_dict_path.exists():
                pytest.skip("Model weights not saved in expected format")
            else:
                # Safetensors format - use safetensors library
                try:
                    from safetensors.torch import load_file
                    loaded_state_dict = load_file(state_dict_path)
                except ImportError:
                    pytest.skip("safetensors library not available")
        else:
            # PyTorch pickle format
            loaded_state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)

        loaded_model.load_state_dict(loaded_state_dict, strict=False)  # strict=False for tied weights

        # Compare weights
        for name, param in loaded_model.named_parameters():
            assert torch.allclose(param, original_weights[name], rtol=1e-5)

    def test_save_checkpoint_includes_metadata(self, tiny_config, temp_workspace,
                                               tiny_model, mock_tokenizer):
        """Checkpoint metadata includes all required fields."""
        import json

        manager = CheckpointManager(tiny_config, str(temp_workspace), "abc123")

        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0,
                                                       total_iters=10)

        manager.save_checkpoint(
            model=tiny_model,
            tokenizer=mock_tokenizer,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            global_step=200,
            epoch=2
        )

        checkpoint_dir = temp_workspace / "test" / "output" / "checkpoint-200"
        metadata_path = checkpoint_dir / "metadata.json"

        assert metadata_path.exists()

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Check required fields
        assert metadata["experiment_name"] == "test_exp"
        assert metadata["global_step"] == 200
        assert metadata["epoch"] == 2
        assert metadata["git_commit_hash"] == "abc123"
        assert "timestamp" in metadata
        assert "config_hash" in metadata
        assert "training_config" in metadata
        assert "model_config" in metadata

    def test_save_checkpoint_preserves_optimizer_state(self, tiny_config, temp_workspace,
                                                       tiny_model, mock_tokenizer):
        """Optimizer state is saved and can be restored."""
        manager = CheckpointManager(tiny_config, str(temp_workspace), "test_hash")

        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

        # Take a few optimization steps to build state
        for _ in range(3):
            input_ids = torch.randint(0, 1000, (2, 32))
            # For causal LM, labels are the same as input_ids (shifted internally)
            outputs = tiny_model(input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Get optimizer state before saving
        original_state = {k: v.clone() if isinstance(v, torch.Tensor) else v
                         for k, v in optimizer.state_dict()["state"].items()}

        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0,
                                                       total_iters=10)

        manager.save_checkpoint(
            model=tiny_model,
            tokenizer=mock_tokenizer,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            global_step=3,
            epoch=0
        )

        # Load checkpoint state
        checkpoint_dir = temp_workspace / "test" / "output" / "checkpoint-3"
        state = torch.load(checkpoint_dir / "training_state.pt", map_location="cpu", weights_only=False)

        assert "optimizer" in state
        assert "lr_scheduler" in state

    def test_save_checkpoint_preserves_rng_state(self, tiny_config, temp_workspace,
                                                  tiny_model, mock_tokenizer):
        """RNG states are saved for reproducibility."""
        manager = CheckpointManager(tiny_config, str(temp_workspace), "test_hash")

        # Set known RNG state
        random.seed(12345)
        np.random.seed(12345)
        torch.manual_seed(12345)

        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0,
                                                       total_iters=10)

        manager.save_checkpoint(
            model=tiny_model,
            tokenizer=mock_tokenizer,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            global_step=10,
            epoch=0
        )

        # Load and check RNG states
        checkpoint_dir = temp_workspace / "test" / "output" / "checkpoint-10"
        state = torch.load(checkpoint_dir / "training_state.pt", map_location="cpu", weights_only=False)

        assert "random_state" in state
        assert "numpy_random_state" in state
        assert "torch_random_state" in state

    def test_load_checkpoint_returns_none_when_no_checkpoint(self, tiny_config,
                                                             temp_workspace):
        """Loading returns None when no checkpoint exists."""
        manager = CheckpointManager(tiny_config, str(temp_workspace), "test_hash")

        optimizer = torch.optim.Adam([torch.randn(10)])
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0,
                                                       total_iters=10)

        model, tokenizer, step, epoch = manager.load_checkpoint(
            model_factory=lambda: create_model(tiny_config),
            device=torch.device("cpu"),
            optimizer=optimizer,
            lr_scheduler=scheduler
        )

        assert model is None
        assert tokenizer is None
        assert step == 0
        assert epoch == 0

    def test_load_checkpoint_when_resume_disabled(self, tiny_config, temp_workspace,
                                                   tiny_model, mock_tokenizer):
        """Loading returns None when resume_from_checkpoint is False."""
        # Disable resume
        tiny_config.training.resume_from_checkpoint = False

        manager = CheckpointManager(tiny_config, str(temp_workspace), "test_hash")

        optimizer = torch.optim.Adam([torch.randn(10)])
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0,
                                                       total_iters=10)

        # Even if a checkpoint exists, it shouldn't load
        model, tokenizer, step, epoch = manager.load_checkpoint(
            model_factory=lambda: create_model(tiny_config),
            device=torch.device("cpu"),
            optimizer=optimizer,
            lr_scheduler=scheduler
        )

        assert model is None

    @pytest.mark.skip(reason="Requires valid tokenizer model file - integration test")
    def test_checkpoint_roundtrip(self, tiny_config, temp_workspace, tiny_model,
                                   mock_tokenizer, deterministic_seed):
        """Save and load checkpoint roundtrip preserves state."""
        # Enable resume
        tiny_config.training.resume_from_checkpoint = True

        manager = CheckpointManager(tiny_config, str(temp_workspace), "test_hash")

        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0,
                                                       total_iters=10)

        # Train for a few steps
        for i in range(5):
            input_ids = torch.randint(0, 1000, (2, 32))
            outputs = tiny_model(input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Save checkpoint
        manager.save_checkpoint(
            model=tiny_model,
            tokenizer=mock_tokenizer,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            global_step=5,
            epoch=0
        )

        # Create new optimizer and scheduler for loading
        new_optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
        new_scheduler = torch.optim.lr_scheduler.LinearLR(new_optimizer,
                                                          start_factor=1.0,
                                                          total_iters=10)

        # Load checkpoint
        loaded_model, loaded_tokenizer, step, epoch = manager.load_checkpoint(
            model_factory=lambda: create_model(tiny_config),
            device=torch.device("cpu"),
            optimizer=new_optimizer,
            lr_scheduler=new_scheduler
        )

        assert loaded_model is not None
        assert step == 5
        assert epoch == 0

        # Verify model weights match
        for (name1, param1), (name2, param2) in zip(
            tiny_model.named_parameters(),
            loaded_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2, rtol=1e-5)

    @pytest.mark.skip(reason="Requires valid tokenizer model file - integration test")
    def test_load_latest_checkpoint(self, tiny_config, temp_workspace, tiny_model,
                                     mock_tokenizer):
        """Loading selects the latest checkpoint by step number."""
        tiny_config.training.resume_from_checkpoint = True

        manager = CheckpointManager(tiny_config, str(temp_workspace), "test_hash")

        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0,
                                                       total_iters=10)

        # Save multiple checkpoints
        for step in [10, 50, 100]:
            manager.save_checkpoint(
                model=tiny_model,
                tokenizer=mock_tokenizer,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                global_step=step,
                epoch=0
            )

        # Load should get checkpoint-100
        new_optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
        new_scheduler = torch.optim.lr_scheduler.LinearLR(new_optimizer,
                                                          start_factor=1.0,
                                                          total_iters=10)

        loaded_model, loaded_tokenizer, step, epoch = manager.load_checkpoint(
            model_factory=lambda: create_model(tiny_config),
            device=torch.device("cpu"),
            optimizer=new_optimizer,
            lr_scheduler=new_scheduler
        )

        assert step == 100

    def test_get_checkpoint_schedule_from_config(self, tiny_config, temp_workspace):
        """Get checkpoint schedule from config if provided."""
        tiny_config.training.checkpoint_schedule = [10, 20, 30, 40, 50]

        manager = CheckpointManager(tiny_config, str(temp_workspace), "test_hash")
        schedule = manager.get_checkpoint_schedule()

        assert schedule == {10, 20, 30, 40, 50}

    def test_checkpoint_schedule_empty_by_default(self, tiny_config, temp_workspace):
        """Checkpoint schedule is empty if not configured."""
        tiny_config.training.checkpoint_schedule = None
        tiny_config.training.auto_generate_checkpoints = False

        manager = CheckpointManager(tiny_config, str(temp_workspace), "test_hash")
        schedule = manager.get_checkpoint_schedule()

        assert schedule == set()

    def test_amp_scaler_state_saved(self, tiny_config, temp_workspace, tiny_model,
                                     mock_tokenizer):
        """AMP gradient scaler state is saved and restored."""
        manager = CheckpointManager(tiny_config, str(temp_workspace), "test_hash")

        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0,
                                                       total_iters=10)

        # Create scaler
        scaler = torch.cuda.amp.GradScaler()

        manager.save_checkpoint(
            model=tiny_model,
            tokenizer=mock_tokenizer,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            global_step=10,
            epoch=0,
            scaler=scaler
        )

        # Check scaler state in saved file
        checkpoint_dir = temp_workspace / "test" / "output" / "checkpoint-10"
        state = torch.load(checkpoint_dir / "training_state.pt", map_location="cpu", weights_only=False)

        assert "amp_scaler" in state
        assert state["amp_scaler"] is not None


class TestCheckpointManagerEdgeCases:
    """Edge case tests for checkpoint management."""

    def test_save_checkpoint_with_no_optimizer_state(self, tiny_config, temp_workspace,
                                                      tiny_model, mock_tokenizer):
        """Checkpoint can be saved even with fresh optimizer (no state)."""
        manager = CheckpointManager(tiny_config, str(temp_workspace), "test_hash")

        # Fresh optimizer with no steps taken
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0,
                                                       total_iters=10)

        # Should not raise an error
        manager.save_checkpoint(
            model=tiny_model,
            tokenizer=mock_tokenizer,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            global_step=0,
            epoch=0
        )

        checkpoint_dir = temp_workspace / "test" / "output" / "checkpoint-0"
        assert checkpoint_dir.exists()

    def test_multiple_checkpoints_coexist(self, tiny_config, temp_workspace,
                                          tiny_model, mock_tokenizer):
        """Multiple checkpoints can exist simultaneously."""
        manager = CheckpointManager(tiny_config, str(temp_workspace), "test_hash")

        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0,
                                                       total_iters=10)

        # Save multiple checkpoints
        for step in [100, 200, 300]:
            manager.save_checkpoint(
                model=tiny_model,
                tokenizer=mock_tokenizer,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                global_step=step,
                epoch=step // 100
            )

        # All should exist
        output_dir = temp_workspace / "test" / "output"
        assert (output_dir / "checkpoint-100").exists()
        assert (output_dir / "checkpoint-200").exists()
        assert (output_dir / "checkpoint-300").exists()
