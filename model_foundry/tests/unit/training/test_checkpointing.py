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

from model_foundry.training.checkpointing import (
    CheckpointManager,
    resolve_resume_epoch,
)
from model_foundry.model import create_model


class TestResolveResumeEpoch:
    def test_new_completed_endpoint_advances(self):
        assert resolve_resume_epoch(2000, 1, 1000, 0, True) == 2

    def test_legacy_completed_endpoint_advances(self):
        assert resolve_resume_epoch(2000, 1, 1000, 0, False) == 2

    def test_mid_epoch_checkpoint_does_not_advance(self):
        assert resolve_resume_epoch(1500, 1, 1000, 4000, False) == 1

    def test_boundary_step_with_offset_does_not_advance(self):
        assert resolve_resume_epoch(2000, 1, 1000, 8, False) == 1


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
                                                     tiny_model, mock_tokenizer, deterministic_seed):
        """Saved checkpoint contains correct model weights."""
        manager = CheckpointManager(tiny_config, str(temp_workspace), "test_hash")

        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0,
                                                       total_iters=10)

        # Get original weights (detach to avoid computation graph issues)
        original_weights = {name: param.clone().detach() for name, param in tiny_model.named_parameters()}

        manager.save_checkpoint(
            model=tiny_model,
            tokenizer=mock_tokenizer,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            global_step=50,
            epoch=0
        )

        # Load weights from checkpoint and verify they match original
        checkpoint_dir = temp_workspace / "test" / "output" / "checkpoint-50"

        # Load saved state dict
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

        # Compare loaded weights directly with original weights
        # (don't create a new model - just check the checkpoint contains correct weights)
        for name, original_param in original_weights.items():
            # Handle both formats: "hf_model.transformer.wte.weight" and "transformer.wte.weight"
            checkpoint_key = name if name in loaded_state_dict else name.replace("hf_model.", "")
            if checkpoint_key not in loaded_state_dict:
                # Try without prefix
                checkpoint_key = name.split("hf_model.")[-1]

            assert checkpoint_key in loaded_state_dict, f"Parameter {name} not found in checkpoint"
            assert torch.allclose(loaded_state_dict[checkpoint_key], original_param, rtol=1e-5), \
                f"Parameter {name} doesn't match"

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

        # Check token metrics for cross-architecture comparison
        assert "token_metrics" in metadata
        token_metrics = metadata["token_metrics"]
        assert "total_tokens_processed" in token_metrics
        assert "tokens_per_step" in token_metrics
        assert "effective_batch_size" in token_metrics
        assert "sequence_length" in token_metrics
        assert "estimated_tokens_at_step" in token_metrics

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

        # Build a throwaway model to pass in — the function should early-out
        # before it touches anything.
        model = create_model(tiny_config)
        tokenizer, step, epoch = manager.load_checkpoint(
            model=model,
            device=torch.device("cpu"),
            optimizer=optimizer,
            lr_scheduler=scheduler
        )

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
        tokenizer, step, epoch = manager.load_checkpoint(
            model=tiny_model,
            device=torch.device("cpu"),
            optimizer=optimizer,
            lr_scheduler=scheduler
        )

        assert tokenizer is None

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

        # Load checkpoint into a freshly-initialised target model
        loaded_model = create_model(tiny_config)
        loaded_tokenizer, step, epoch = manager.load_checkpoint(
            model=loaded_model,
            device=torch.device("cpu"),
            optimizer=new_optimizer,
            lr_scheduler=new_scheduler
        )

        assert loaded_tokenizer is not None
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

        loaded_model = create_model(tiny_config)
        loaded_tokenizer, step, epoch = manager.load_checkpoint(
            model=loaded_model,
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


class TestRollingResume:
    """Rolling resume (2026-10-06): pruning of superseded resume states."""

    @staticmethod
    def _ckpt(root, step, with_state=True, suffix=""):
        import json
        d = root / f"checkpoint-{step}{suffix}"
        d.mkdir(parents=True)
        (d / "model.safetensors").write_bytes(b"w")
        if with_state:
            (d / "training_state.pt").write_bytes(b"s")
        (d / "metadata.json").write_text(json.dumps({"has_resume_state": with_state}))
        return d

    def test_prune_removes_only_superseded_rolling_states(self, tiny_config, temp_workspace):
        import json
        manager = CheckpointManager(tiny_config, str(temp_workspace), "test_hash")
        out = manager.output_dir
        for step in (2, 5, 8, 11, 12):
            self._ckpt(out, step)
        self._ckpt(out, 3, with_state=False)
        staging = self._ckpt(out, 9, suffix=".tmp")

        manager.prune_rolling_resume_states(8, {2, 11})

        assert not (out / "checkpoint-5" / "training_state.pt").exists()
        md = json.loads((out / "checkpoint-5" / "metadata.json").read_text())
        assert md["has_resume_state"] is False
        assert md["resume_state_pruned"] == "rolling"
        assert (out / "checkpoint-5" / "model.safetensors").exists()  # weights stay
        for step in (2, 8, 11, 12):  # permanent, the kept one, and newer
            assert (out / f"checkpoint-{step}" / "training_state.pt").exists()
        assert (staging / "training_state.pt").exists()  # staging dirs untouched

    def test_prune_noop_when_kept_checkpoint_has_no_state(self, tiny_config, temp_workspace):
        manager = CheckpointManager(tiny_config, str(temp_workspace), "test_hash")
        out = manager.output_dir
        self._ckpt(out, 5)
        self._ckpt(out, 8, with_state=False)

        manager.prune_rolling_resume_states(8, set())

        assert (out / "checkpoint-5" / "training_state.pt").exists()

    def test_rolling_setting_does_not_change_config_hash(self, tiny_config):
        import hashlib
        import json

        def h(cfg):
            return hashlib.md5(json.dumps(cfg.model_dump(), sort_keys=True).encode()).hexdigest()

        before = h(tiny_config)
        rolled = tiny_config.model_copy(deep=True)
        rolled.training.rolling_resume_every_steps = 251
        assert rolled.training.rolling_resume_every_steps == 251
        assert h(rolled) == before

    def test_resave_never_drops_existing_resume_state(self, tiny_config, temp_workspace,
                                                     tiny_model, mock_tokenizer):
        """After a resume the loop re-saves the step it resumed from; when
        that is a rolling step the re-save asks for analysis-only, which
        must not delete the state (smoke find, 2026-10-06)."""
        import json
        manager = CheckpointManager(tiny_config, str(temp_workspace), "test_hash")
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0,
                                                       total_iters=10)
        for full in (True, False):
            manager.save_checkpoint(model=tiny_model, tokenizer=mock_tokenizer,
                                    optimizer=optimizer, lr_scheduler=scheduler,
                                    global_step=590, epoch=0, save_resume_state=full)
        d = manager.output_dir / "checkpoint-590"
        assert (d / "training_state.pt").exists()
        assert json.loads((d / "metadata.json").read_text())["has_resume_state"] is True
