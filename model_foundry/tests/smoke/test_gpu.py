"""
GPU smoke tests for the training pipeline.

These tests verify GPU-specific behavior that can only be validated on CUDA:
- AMP (automatic mixed precision) training path
- GradScaler overflow handling
- Flash Attention fallback
- Deterministic cuDNN settings
- CUDA memory management
- Checkpoint save/restore round-trip on GPU
- Multi-architecture forward/backward on GPU

All tests are marked with @pytest.mark.gpu and skipped when CUDA is unavailable.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from model_foundry.config import (
    ExperimentConfig, DataConfig, TokenizerConfig, ModelConfig,
    TransformerModelConfig, TrainingConfig, LoggingConfig
)

# Skip entire module if CUDA is not available
pytestmark = pytest.mark.gpu

CUDA_AVAILABLE = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")


def _make_gpu_config(architecture="gpt2", objective="causal_lm", use_amp=True,
                     gradient_accumulation_steps=1):
    """Create a minimal config suitable for GPU smoke testing."""
    if architecture in ("gpt2", "bert"):
        model_cfg = ModelConfig(
            architecture=architecture,
            transformer=TransformerModelConfig(
                layers=2,
                embedding_size=64,
                hidden_size=64,
                intermediate_hidden_size=128,
                attention_heads=2,
                activation_function="gelu",
                dropout=0.1,
                attention_dropout=0.1,
            ),
        )
    elif architecture in ("lstm", "gru", "rnn"):
        from model_foundry.config import RNNModelConfig
        model_cfg = ModelConfig(
            architecture=architecture,
            rnn=RNNModelConfig(
                embedding_size=64,
                hidden_size=128,
                num_layers=2,
                bidirectional=False,
                dropout=0.1,
                rnn_type=architecture,
            ),
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return ExperimentConfig(
        experiment_name=f"gpu_smoke_{architecture}",
        data=DataConfig(
            source_corpus="test/data",
            training_corpus="test/data/train",
            batch_size=4,
            max_sequence_length=32,
        ),
        tokenizer=TokenizerConfig(output_dir="test/tokenizer", vocab_size=1000),
        model=model_cfg,
        training=TrainingConfig(
            output_dir="test/output",
            learning_rate=1e-4,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            epochs=1,
            train_steps=10,
            warmup_steps=2,
            objective=objective,
            use_amp=use_amp,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=1.0,
            use_tf32=True,
            use_gradient_checkpointing=False,
        ),
        logging=LoggingConfig(level="WARNING", dir="test/logs", use_wandb=False),
        random_seed=42,
    )


# =====================================================================
# AMP & GradScaler
# =====================================================================

class TestAMPOnGPU:
    """Verify automatic mixed precision works end-to-end on a real GPU."""

    @skip_no_cuda
    def test_amp_forward_backward(self):
        """AMP forward/backward produces finite gradients on GPU."""
        from model_foundry.model import create_model

        config = _make_gpu_config(use_amp=True)
        model = create_model(config).to("cuda")

        scaler = torch.amp.GradScaler("cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        input_ids = torch.randint(0, 1000, (4, 32), device="cuda")
        labels = torch.randint(0, 1000, (4, 32), device="cuda")

        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(input_ids, labels=labels)
            loss = output.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        assert torch.isfinite(loss), "AMP loss is not finite"
        # Verify at least some params have gradients
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, "No parameters received gradients under AMP"

    @skip_no_cuda
    def test_amp_disabled_uses_fp32(self):
        """With AMP disabled, forward pass stays in fp32."""
        from model_foundry.model import create_model

        config = _make_gpu_config(use_amp=False)
        model = create_model(config).to("cuda")

        input_ids = torch.randint(0, 1000, (4, 32), device="cuda")
        output = model(input_ids)

        assert output.logits.dtype == torch.float32, \
            f"Expected fp32 logits without AMP, got {output.logits.dtype}"

    @skip_no_cuda
    def test_gradscaler_handles_overflow(self):
        """GradScaler correctly handles inf gradients without crashing."""
        model = nn.Linear(64, 64).cuda()
        scaler = torch.amp.GradScaler("cuda")
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

        # Manufacture an overflow
        x = torch.randn(4, 64, device="cuda") * 1e6
        with torch.amp.autocast("cuda", dtype=torch.float16):
            y = model(x)
            loss = y.sum()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Should not crash — scaler skips the step on overflow
        scaler.step(optimizer)
        scaler.update()


# =====================================================================
# Deterministic cuDNN
# =====================================================================

class TestDeterministicCUDA:
    """Verify our deterministic settings produce reproducible results."""

    @skip_no_cuda
    def test_cudnn_deterministic_flag(self):
        """After Trainer._setup_memory_management, cuDNN is deterministic."""
        from model_foundry.trainer import Trainer

        config = _make_gpu_config()
        # Temporarily create a Trainer (it calls _setup_memory_management in __init__)
        with patch("model_foundry.trainer.create_data_processor"):
            trainer = Trainer(config, "/tmp")

        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    @skip_no_cuda
    def test_same_seed_same_output(self):
        """Same seed + deterministic mode → identical outputs on GPU."""
        from model_foundry.model import create_model
        from model_foundry.utils import set_seed

        config = _make_gpu_config()
        input_ids = torch.randint(0, 1000, (4, 32), device="cuda")

        set_seed(42)
        model1 = create_model(config).to("cuda")
        model1.eval()
        with torch.no_grad():
            out1 = model1(input_ids).logits

        set_seed(42)
        model2 = create_model(config).to("cuda")
        model2.eval()
        with torch.no_grad():
            out2 = model2(input_ids).logits

        assert torch.allclose(out1, out2, atol=1e-5), \
            "Same seed did not produce identical GPU outputs"


# =====================================================================
# Flash Attention fallback
# =====================================================================

class TestFlashAttention:
    """Verify Flash Attention fallback logic."""

    @skip_no_cuda
    def test_model_creates_with_or_without_flash(self):
        """Model creation succeeds regardless of Flash Attention availability."""
        from model_foundry.model import create_model

        config = _make_gpu_config()

        # Try with flash attention request
        try:
            model = create_model(config, attn_implementation="flash_attention_2")
        except (ImportError, ValueError):
            # Fallback — should still work
            model = create_model(config)

        model = model.to("cuda")
        input_ids = torch.randint(0, 1000, (2, 16), device="cuda")
        output = model(input_ids)
        assert output.logits is not None


# =====================================================================
# Multi-architecture GPU forward/backward
# =====================================================================

class TestMultiArchGPU:
    """Run all architectures through forward/backward on GPU."""

    @skip_no_cuda
    @pytest.mark.parametrize("arch,objective", [
        ("gpt2", "causal_lm"),
        ("bert", "masked_lm"),
        ("lstm", "causal_lm"),
        ("gru", "causal_lm"),
    ])
    def test_forward_backward_on_gpu(self, arch, objective):
        """Architecture produces finite loss and gradients on GPU."""
        from model_foundry.model import create_model

        config = _make_gpu_config(architecture=arch, objective=objective)
        model = create_model(config).to("cuda")

        input_ids = torch.randint(0, 1000, (4, 32), device="cuda")
        labels = torch.randint(0, 1000, (4, 32), device="cuda")
        if objective == "masked_lm":
            labels[:, :5] = -100

        output = model(input_ids, labels=labels)
        assert torch.isfinite(output.loss), f"{arch}: loss is not finite on GPU"

        output.loss.backward()
        emb = model.get_input_embeddings()
        assert emb.weight.grad is not None, f"{arch}: no gradient on GPU"
        assert not torch.isnan(emb.weight.grad).any(), f"{arch}: NaN gradients on GPU"

    @skip_no_cuda
    @pytest.mark.parametrize("arch", ["gpt2", "lstm"])
    def test_amp_training_step_on_gpu(self, arch):
        """Full AMP micro-batch step works for each architecture on GPU."""
        from model_foundry.model import create_model

        config = _make_gpu_config(architecture=arch, use_amp=True,
                                  gradient_accumulation_steps=2)
        model = create_model(config).to("cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")

        for micro_step in range(4):
            input_ids = torch.randint(0, 1000, (4, 32), device="cuda")
            labels = torch.randint(0, 1000, (4, 32), device="cuda")

            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = model(input_ids, labels=labels)
                loss = output.loss / 2  # grad accumulation

            scaler.scale(loss).backward()

            if (micro_step + 1) % 2 == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        # Should complete 2 optimizer steps without error
        assert torch.isfinite(loss)


# =====================================================================
# Checkpoint round-trip on GPU
# =====================================================================

class TestCheckpointGPU:
    """Verify checkpoint save/load round-trip preserves GPU state."""

    @skip_no_cuda
    def test_checkpoint_round_trip(self):
        """Save and restore a model on GPU, verify identical outputs."""
        from model_foundry.model import create_model
        from model_foundry.utils import set_seed

        config = _make_gpu_config()
        set_seed(42)
        model = create_model(config).to("cuda")
        model.eval()

        input_ids = torch.randint(0, 1000, (2, 16), device="cuda")
        with torch.no_grad():
            original_out = model(input_ids).logits.cpu()

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            # Reload — HF saves as safetensors by default
            set_seed(42)
            model2 = create_model(config).to("cuda")
            safetensors_path = Path(tmpdir) / "model.safetensors"
            bin_path = Path(tmpdir) / "pytorch_model.bin"
            if safetensors_path.exists():
                from safetensors.torch import load_file
                state = load_file(str(safetensors_path), device="cuda")
            else:
                state = torch.load(bin_path, map_location="cuda",
                                   weights_only=False)
            model2.load_state_dict(state)
            model2.eval()

            with torch.no_grad():
                restored_out = model2(input_ids).logits.cpu()

        assert torch.allclose(original_out, restored_out, atol=1e-5), \
            "Checkpoint round-trip changed model outputs"


# =====================================================================
# CUDA memory management
# =====================================================================

class TestCUDAMemory:
    """Verify CUDA memory management settings."""

    @skip_no_cuda
    def test_tf32_settings_applied(self):
        """TF32 settings are applied after Trainer init (Ampere+ GPUs only)."""
        from model_foundry.trainer import Trainer

        config = _make_gpu_config()
        with patch("model_foundry.trainer.create_data_processor"):
            Trainer(config, "/tmp")

        # TF32 is only supported on Ampere+ (compute capability >= 8.0).
        # On older GPUs, PyTorch silently ignores the setting.
        cc_major = torch.cuda.get_device_capability()[0]
        if cc_major >= 8:
            assert torch.backends.cuda.matmul.allow_tf32 is True
            assert torch.backends.cudnn.allow_tf32 is True
        else:
            # On pre-Ampere, just verify no crash occurred — TF32 is a no-op
            pass

    @skip_no_cuda
    def test_empty_cache_does_not_error(self):
        """torch.cuda.empty_cache() runs without error (basic sanity)."""
        t = torch.randn(1000, 1000, device="cuda")
        del t
        torch.cuda.empty_cache()
        # Should not raise
