"""
Main Trainer class for the Model Foundry framework.

This module coordinates the training process by orchestrating the model,
data processing, checkpointing, and training loop components.
"""

import argparse
import os
import sys
import yaml
import logging
import subprocess
from pathlib import Path
import torch
from torch.optim import AdamW
from transformers import get_scheduler
import wandb

# Import refactored components
from .config import ExperimentConfig
from .model import create_model
from .utils import find_project_root, set_seed, get_device, get_git_commit_hash
from .data import create_data_processor
from .logging_utils import setup_logging
from .training import CheckpointManager, load_tokenizer, TrainingLoop
from .training.checkpointing import resolve_resume_epoch


class Trainer:
    """
    Main trainer class that coordinates the training process.

    This class sets up the training environment, initializes all components,
    and orchestrates the training workflow.
    """

    def __init__(self, config: ExperimentConfig, base_dir: str):
        """
        Initialize the trainer.

        Args:
            config: Validated experiment configuration
            base_dir: Project base directory
        """
        self.config = config
        self.base_dir = base_dir
        self.device = get_device()
        self.git_commit_hash = get_git_commit_hash()

        # Components to be initialized
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.dataloader = None
        self.tokenizer = None
        self.checkpoint_manager = None
        self.training_loop = None

        # Initialize data processor
        self.data_processor = create_data_processor(config, base_dir)

        # Setup memory management
        self._setup_memory_management()

    def _registry_identity_for_wandb(self):
        """Extract the (run_id, arch, lang, condition, seed, run_kind) dict
        that ``wandb_init.init_wandb`` expects. Mirrors
        ``cli.py::_registry_identity`` — launchers populate REGISTRY_*
        env vars; hand-invoked runs fall back to defaults so the record
        still goes somewhere sensible."""
        arch = self.config.model.architecture
        lang = os.environ.get("REGISTRY_LANG") or getattr(self.config, "lang", None) or "unknown"
        condition = os.environ.get("REGISTRY_CONDITION") or getattr(self.config, "condition", None) or "unknown"
        seed_env = os.environ.get("REGISTRY_SEED")
        seed = int(seed_env) if seed_env is not None else int(getattr(self.config, "seed", None) or self.config.random_seed)
        run_id = (
            os.environ.get("REGISTRY_RUN_ID")
            or getattr(self.config, "run_id", None)
            or self.config.experiment_name
        )
        run_kind = os.environ.get("REGISTRY_RUN_KIND", "production")
        return {
            "run_id": run_id, "arch": arch, "lang": lang,
            "condition": condition, "seed": seed, "run_kind": run_kind,
        }

    def _setup_memory_management(self):
        """Configure CUDA memory management settings.

        Begins with a fail-fast GPU health probe. On NRP's shared GPU
        nodes a stale CUDA context from another tenant occasionally
        renders the assigned device "busy" for our pod's lifetime. Every
        downstream torch.cuda call then fails with the same opaque
        "CUDA-capable device(s) is/are busy or unavailable". Catching
        this here lets us exit with code 2 (specific signal: hardware
        transient) so the K8s Job creates a replacement pod on a
        different node, rather than letting the trial limp through
        wandb-init then dying mid-trainer-setup.
        """
        if not torch.cuda.is_available():
            return

        try:
            torch.cuda.synchronize()
            # Force a real allocation + kernel launch — exercises both
            # the memory allocator and the compute path. A stale context
            # from another tenant fails one of these even when
            # torch.cuda.is_available() returned True.
            _probe = torch.zeros(1024, device='cuda')
            _ = _probe.sum().item()
            del _probe
            torch.cuda.empty_cache()
        except RuntimeError as e:
            msg = str(e)
            transient_markers = (
                "CUDA-capable device",
                "all CUDA-capable devices are busy",
                "CUDA error: unknown error",
                "CUDA driver version is insufficient",
            )
            if any(m in msg for m in transient_markers):
                print(
                    f"FATAL: GPU not healthy at trainer init: {msg}",
                    file=sys.stderr,
                )
                print(
                    "  Likely a transient NRP issue (stale CUDA context from a co-tenant",
                    file=sys.stderr,
                )
                print(
                    "  on the same physical GPU). Exiting with code 2 so the K8s Job",
                    file=sys.stderr,
                )
                print(
                    "  creates a replacement pod on a different node.",
                    file=sys.stderr,
                )
                sys.exit(2)
            # Not the transient pattern — re-raise so we get a real traceback.
            raise

        # Set memory fraction to prevent OOM. 0.90 (vs 0.95) leaves a
        # bit of GPU headroom for the driver itself + transient
        # allocations from torch internals; the cost is negligible
        # since our trials are not memory-bound at sampled batch sizes.
        torch.cuda.set_per_process_memory_fraction(0.9)

        # Configure allocator for better performance
        if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ or not os.environ['PYTORCH_CUDA_ALLOC_CONF']:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

        # Deterministic cuDNN selection (conv/RNN algo choice). NOTE the
        # honest scope: this covers ONLY cuDNN ops. cuBLAS GEMMs, FA2's
        # atomicAdd backward, fused AdamW, and Mamba's selective-scan
        # kernels remain run-to-run nondeterministic on this path — the
        # production claim is statistical replication, not bitwise.
        # For bitwise verification runs, set training.deterministic: true
        # (see _configure_determinism).
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self._configure_determinism()

        print("  - GPU healthy + CUDA memory configured (95% limit, "
              "max_split_size_mb:512, deterministic cuDNN algo selection — "
              "see training.deterministic for full bitwise mode)")

    def _configure_determinism(self):
        """Opt-in bitwise-determinism mode (replicability audit, G1).

        Default OFF: production runs keep FA2 + TF32 + nondeterministic
        kernels for speed and claim statistical replication. When
        ``training.deterministic`` is true (the bit-repro verification
        subset):

        - ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` makes cuBLAS GEMMs
          deterministic (must be set before the first cuBLAS call — we
          run before any matmul; pod templates may also set it in env).
        - ``torch.use_deterministic_algorithms(True, warn_only=True)``
          routes every op with a deterministic variant to it and WARNS
          (not raises) on ops without one, so per-arch gaps surface in
          logs instead of crashing the run.
        - TF32 is forced off (overrides ``use_tf32``).
        - Model init forces SDPA instead of FA2 (FA2 2.7.4 has no
          deterministic backward; transformers 4.41.2 exposes no kwarg).
        """
        if not getattr(self.config.training, "deterministic", False):
            return
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        if self.config.training.use_tf32:
            print("  - [deterministic] overriding use_tf32 -> False")
            self.config.training.use_tf32 = False
        print("  - [deterministic] bitwise mode ON: CUBLAS_WORKSPACE_CONFIG="
              f"{os.environ['CUBLAS_WORKSPACE_CONFIG']}, "
              "use_deterministic_algorithms(warn_only), TF32 off, SDPA attention")

    def _calculate_training_parameters(self):
        """Calculate training parameters based on dataset size if not explicitly set."""
        try:
            steps_per_epoch = self.data_processor.get_training_steps_per_epoch()
        except Exception as e:
            print(f"  ⚠️  Could not determine exact dataset size ({e}), using estimates")
            steps_per_epoch = 100
        self.steps_per_epoch = steps_per_epoch

        # Calculate train_steps if not specified
        if self.config.training.train_steps is None:
            calculated_train_steps = steps_per_epoch * self.config.training.epochs
            print(f"  - Auto-calculated train_steps: {calculated_train_steps} "
                  f"({self.config.training.epochs} epochs × {steps_per_epoch} steps/epoch)")
            self.config.training.train_steps = calculated_train_steps

        scheduler_train_steps = (
            self.config.training.scheduler_train_steps
            or self.config.training.train_steps
        )
        if scheduler_train_steps < self.config.training.train_steps:
            raise ValueError(
                "scheduler_train_steps cannot be shorter than train_steps"
            )
        self.config.training.scheduler_train_steps = scheduler_train_steps

        # Calculate warmup_steps if not specified
        if self.config.training.warmup_steps is None:
            calculated_warmup_steps = int(
                scheduler_train_steps * self.config.training.warmup_ratio
            )
            print(f"  - Auto-calculated warmup_steps: {calculated_warmup_steps} "
                  f"({self.config.training.warmup_ratio:.1%} of scheduler horizon)")
            self.config.training.warmup_steps = calculated_warmup_steps

        print(f"  - Final training configuration:")
        print(f"    - Epochs: {self.config.training.epochs}")
        print(f"    - Steps per epoch: {steps_per_epoch}")
        print(f"    - Total training steps: {self.config.training.train_steps}")
        print(f"    - Scheduler horizon: {scheduler_train_steps}")
        print(f"    - Warmup steps: {self.config.training.warmup_steps}")

    def _prepare_data(self):
        """Load and prepare the dataset and dataloader."""
        if not self.data_processor.preprocess_data():
            raise RuntimeError("Data preprocessing failed")

        self.dataloader = self.data_processor.create_dataloader(self.tokenizer)

    def _initialize_model(self):
        """Initialize the model with appropriate optimizations."""
        # Tolerate Dynamo graph-capture failures by falling back to eager
        # for the offending graph instead of crashing. (A previous bare
        # `torch._dynamo.disable()` call here was a no-op — without a
        # function argument it just returns a decorator — and was removed
        # for clarity; compile is governed by training.compile_mode.)
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True

        # Bitwise-deterministic runs force SDPA: FA2 2.7.4's backward is
        # atomicAdd-nondeterministic and transformers 4.41.2 exposes no
        # deterministic kwarg for it (replicability audit, G1).
        if getattr(self.config.training, "deterministic", False):
            print("  - [deterministic] forcing SDPA attention (skipping FA2)")
            try:
                self.model = create_model(
                    self.config,
                    attn_implementation="sdpa"
                ).to(self.device)
                print("  - Successfully initialized model with PyTorch SDPA")
            except (ImportError, ValueError, TypeError) as e2:
                print(f"  - SDPA unavailable ({e2}); falling back to eager attention")
                self.model = create_model(self.config).to(self.device)
                print("  - Model created with eager (standard) attention")
            self._apply_torch_compile()
            self._apply_memory_optimizations()
            return

        try:
            # Try Flash Attention 2 first (Ampere+ GPUs, supported by GPT-2 / Mamba)
            print("  - Attempting to create model with Flash Attention 2...")
            self.model = create_model(
                self.config,
                attn_implementation="flash_attention_2"
            ).to(self.device)
            print("  - Successfully initialized model with Flash Attention 2")
        except (ImportError, ValueError) as e:
            # FA2 unavailable — try PyTorch SDPA before giving up to eager.
            # BertForMaskedLM (and a few others) doesn't accept FA2 in
            # transformers 4.41, but supports SDPA from 4.36+. SDPA uses
            # memory-efficient attention internally and is typically
            # 1.5-2x faster than the eager (standard) attention path at
            # seq_len=1000.
            print(f"  - Flash Attention 2 not available ({e}); trying PyTorch SDPA")
            try:
                self.model = create_model(
                    self.config,
                    attn_implementation="sdpa"
                ).to(self.device)
                print("  - Successfully initialized model with PyTorch SDPA")
            except (ImportError, ValueError, TypeError) as e2:
                print(f"  - SDPA also unavailable ({e2}); falling back to eager attention")
                self.model = create_model(self.config).to(self.device)
                print("  - Model created with eager (standard) attention")

        # Apply torch.compile if configured
        self._apply_torch_compile()

        # Apply memory optimizations
        self._apply_memory_optimizations()

    def _apply_torch_compile(self):
        """Apply torch.compile optimization if configured."""
        compile_mode = self.config.training.compile_mode

        if not compile_mode or str(compile_mode).lower() == 'none':
            print(f"  - Skipping torch.compile (compile_mode='{compile_mode}')")
            return

        try:
            print(f"  - Compiling model with torch.compile(mode='{compile_mode}')...")
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            # 1.3 — let Dynamo accumulate more compiled graphs before falling
            # back to eager. Long runs otherwise trip the default cache size
            # (8) and silently revert to unoptimized execution.
            torch._dynamo.config.cache_size_limit = 256

            # 1.3 — drop the hard-coded backend="eager" that defeated
            # torch.compile's speedup; the default backend (Inductor) is
            # where the win lives. Gated behind compile_mode in config so
            # archs that regress under Inductor can opt out per-YAML.
            if compile_mode == 'reduce-overhead':
                self.model = torch.compile(self.model, mode="reduce-overhead")
            elif compile_mode == 'max-autotune':
                self.model = torch.compile(self.model, mode="max-autotune")
            else:
                self.model = torch.compile(self.model, mode="default")

            print("  - Model compilation successful (backend=inductor)")
        except Exception as e:
            print(f"  - Warning: torch.compile failed ({e}), continuing without compilation")

    def _apply_memory_optimizations(self):
        """Apply memory optimization settings."""
        if self.config.training.use_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("  - TF32 enabled for faster training on Ampere+ GPUs")

        if self.config.training.use_gradient_checkpointing:
            # Different architectures expose gradient checkpointing under
            # different method names. Try the common ones in order.
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("  - Gradient checkpointing enabled to save memory")
            elif hasattr(self.model, 'enable_gradient_checkpointing'):
                self.model.enable_gradient_checkpointing()
                print("  - Gradient checkpointing enabled to save memory")
            else:
                print(f"  - WARNING: gradient checkpointing requested but "
                      f"{type(self.model).__name__} does not expose it")

    def _initialize_optimizer_and_scheduler(self):
        """Initialize optimizer and learning rate scheduler."""
        # 0.5 — use fused AdamW on CUDA. ~10–20% faster on the optimizer
        # step; requires contiguous params on the same device (true for
        # our single-GPU training). CPU builds of PyTorch don't support
        # fused=True, so gate on device type and fall back gracefully.
        use_fused = torch.cuda.is_available() and self.device.type == "cuda"
        optimizer_kwargs = dict(
            lr=self.config.training.learning_rate,
            betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
            eps=self.config.training.adam_epsilon,
        )
        try:
            self.optimizer = AdamW(
                self.model.parameters(),
                fused=use_fused,
                **optimizer_kwargs,
            )
            if use_fused:
                print("  - Using fused AdamW optimizer")
        except (RuntimeError, TypeError) as e:
            # Older torch versions reject `fused=` entirely; also a small
            # number of param configurations can raise at construction.
            print(f"  - Fused AdamW unavailable ({e}); using standard AdamW")
            self.optimizer = AdamW(self.model.parameters(), **optimizer_kwargs)

        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=(
                self.config.training.scheduler_train_steps
                or self.config.training.train_steps
            )
        )

    def _load_tokenizer(self):
        """Load the tokenizer from its directory."""
        tokenizer_path = os.path.join(self.base_dir, self.config.tokenizer.output_dir)
        self.tokenizer = load_tokenizer(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _sync_vocab_size_with_tokenizer(self):
        """Override config.tokenizer.vocab_size with the tokenizer's actual length.

        `add_special_tokens` in the tokenizer factory can push the real vocab
        size beyond the configured value (e.g. BERT adds [CLS]/[SEP]/[MASK]/
        [PAD]/[UNK] on top of a 50004-piece SentencePiece model, giving a
        true length of 50009). The model's embedding table must match — if
        the tokenizer emits an id >= model vocab_size the forward pass trips
        a CUDA device-side assert (indexSelectLargeIndex: srcIndex <
        srcSelectDimSize).
        """
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            return
        actual = len(self.tokenizer)
        configured = self.config.tokenizer.vocab_size
        if actual != configured:
            print(f"  - Adjusting vocab_size {configured} → {actual} "
                  f"(tokenizer has {actual - configured} extra special tokens)")
            self.config.tokenizer.vocab_size = actual

    def _save_environment_snapshot(self):
        """Save environment snapshot for reproducibility."""
        import datetime

        log_dir = os.path.join(self.base_dir, self.config.logging.dir, self.config.experiment_name)
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        env_file = os.path.join(log_dir, f"env_{timestamp}.txt")

        try:
            with open(env_file, 'w') as f:
                f.write(f"Experiment: {self.config.experiment_name}\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                f.write(f"Git commit: {self.git_commit_hash}\n\n")

                # Save pip freeze
                f.write("=== PIP PACKAGES ===\n")
                try:
                    pip_output = subprocess.check_output(['pip', 'freeze'], text=True)
                    f.write(pip_output)
                except Exception as e:
                    f.write(f"Error getting pip packages: {e}\n")

                f.write("\n=== CUDA INFO ===\n")
                if torch.cuda.is_available():
                    f.write(f"CUDA available: True\n")
                    f.write(f"CUDA version: {torch.version.cuda}\n")
                    f.write(f"GPU count: {torch.cuda.device_count()}\n")
                    for i in range(torch.cuda.device_count()):
                        f.write(f"GPU {i}: {torch.cuda.get_device_name(i)}\n")

                    try:
                        nvidia_output = subprocess.check_output(['nvidia-smi'], text=True)
                        f.write(f"\nNVIDIA-SMI output:\n{nvidia_output}\n")
                    except Exception as e:
                        f.write(f"Error getting NVIDIA-SMI info: {e}\n")
                else:
                    f.write("CUDA available: False\n")

                f.write(f"\n=== SYSTEM INFO ===\n")
                f.write(f"PyTorch version: {torch.__version__}\n")
                f.write(f"Python version: {subprocess.check_output(['python', '--version'], text=True).strip()}\n")

            print(f"  - Environment snapshot saved to: {env_file}")

        except Exception as e:
            print(f"  - Warning: Failed to save environment snapshot: {e}")

    def train(self, on_epoch_end=None):
        """Main training entry point.

        Args:
            on_epoch_end: Optional callable (epoch: int, tokens_seen: int) -> bool
                fired at the end of each epoch. Return True to stop training
                early (used by the sweep agent for patience-based early stop).
        """
        logger = setup_logging(
            "trainer",
            experiment=self.config.experiment_name,
            log_dir=self.config.logging.dir,
            level=getattr(logging, self.config.logging.level)
        )
        logger.info(f"--- Starting Training Run for: {self.config.experiment_name} ---")

        try:
            return self._train_loop(on_epoch_end=on_epoch_end)
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")

            print(f"\n❌ Training failed: {str(e)}")
            print(f"Error type: {type(e).__name__}")

            # CUDA error debugging
            if "CUDA" in str(e) and torch.cuda.is_available():
                print(f"\n🔍 CUDA Error Debugging Information:")
                print(f"   - CUDA Device: {torch.cuda.get_device_name()}")
                print(f"   - Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                print(f"   - Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
                print(f"   - Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

                print(f"\n💡 Debugging suggestions:")
                print(f"   - Try reducing batch_size further (current: {self.config.data.batch_size})")
                print(f"   - Enable gradient checkpointing: use_gradient_checkpointing: true")
                print(f"   - Run with CUDA_LAUNCH_BLOCKING=1 for better error tracing")

            logger.debug("Full traceback:", exc_info=True)
            raise SystemExit(1)

    def _train_loop(self, on_epoch_end=None):
        """Internal training loop implementation."""
        print("Starting training setup...")

        set_seed(self.config.random_seed)

        # Calculate training parameters
        print("  - Calculating training parameters from dataset...")
        self._calculate_training_parameters()

        # Load tokenizer BEFORE model init so we can size the embedding
        # table to match the tokenizer's actual vocab (including any
        # special tokens the tokenizer factory added via
        # `add_special_tokens`, which can push real length beyond the
        # configured vocab_size — e.g. BERT's [CLS]/[SEP]/[MASK]).
        print("  - Loading tokenizer...")
        self._load_tokenizer()
        self._sync_vocab_size_with_tokenizer()

        # Initialize components.
        # Re-seed IMMEDIATELY before model construction so initial weights
        # are a pure function of (arch, seed) — condition-invariant by
        # construction, not by the accident of nothing upstream consuming
        # RNG (replicability audit, G3 / the paired-contrast design).
        # Anything inserted between set_seed and model init in the future
        # can no longer silently break init pairing.
        set_seed(self.config.random_seed)
        print("  - Initializing model...")
        self._initialize_model()

        print("  - Initializing optimizer and scheduler...")
        self._initialize_optimizer_and_scheduler()

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            self.config, self.base_dir, self.git_commit_hash
        )

        # Try to load from checkpoint. We pass self.model in so that weights
        # are loaded in-place — critical because self.optimizer is already
        # bound to self.model.parameters() from _initialize_optimizer_and_scheduler
        # above. Replacing self.model would orphan the optimizer.
        # Diagnostic (2026-06-05 phantom-resume incident): runs have been
        # observed resuming with resume_from_checkpoint=false in their
        # config file. Log the ACTUAL flag value at the call site so any
        # recurrence pinpoints where the flip happens instead of being
        # unexplainable after the fact.
        logger = logging.getLogger("trainer")
        logger.info(
            "load_checkpoint gate: resume_from_checkpoint=%s output_dir=%s",
            self.config.training.resume_from_checkpoint,
            self.checkpoint_manager.output_dir,
        )
        tokenizer, global_step, epoch = self.checkpoint_manager.load_checkpoint(
            model=self.model,
            device=self.device,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler
        )

        # Legacy endpoint checkpoints (written before epoch_completed was
        # persisted) stored offset=0 and the just-completed epoch index.
        # Advance only when the saved step is exactly that epoch boundary;
        # ordinary mid-epoch checkpoints keep the fast-forward path.
        resolved_epoch = resolve_resume_epoch(
            global_step=global_step,
            saved_epoch=epoch,
            steps_per_epoch=self.steps_per_epoch,
            resume_batch_offset=getattr(
                self.checkpoint_manager, "resume_batch_offset", 0
            ),
            epoch_completed=getattr(
                self.checkpoint_manager, "resume_epoch_completed", False
            ),
        )
        if resolved_epoch != epoch:
            logger.warning(
                "Completed endpoint checkpoint detected at step %d: advancing "
                "resume epoch from %d to %d",
                global_step, epoch, resolved_epoch,
            )
            epoch = resolved_epoch
            self.checkpoint_manager.resume_batch_offset = 0
            self.checkpoint_manager.resume_micro_step = 0

        # If a checkpoint-loaded tokenizer exists, prefer it (its vocab
        # should already match the loaded model weights).
        if tokenizer is not None:
            self.tokenizer = tokenizer

        # Prepare data
        print("  - Preparing data...")
        self._prepare_data()

        # Initialize W&B logging via the shared helper (centralizes
        # project/group/job_type/tags so the UI is consistent across
        # cli.py, sweep agents, and any future launcher).
        if self.config.logging.use_wandb:
            from .wandb_init import init_wandb
            # Resume the same WandB run on restart — id lives in the
            # prior checkpoint's metadata.
            resume_id = None
            if global_step > 0:
                import json
                metadata_path = self.checkpoint_manager.output_dir / f"checkpoint-{global_step}" / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    resume_id = metadata.get('wandb_run_id')

            identity = self._registry_identity_for_wandb()
            init_wandb(self.config, identity, resume_id=resume_id)

        # Save environment snapshot
        self._save_environment_snapshot()

        # Initialize training loop
        self.training_loop = TrainingLoop(
            config=self.config,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            dataloader=self.dataloader,
            device=self.device,
            checkpoint_manager=self.checkpoint_manager,
            data_processor=self.data_processor
        )

        # Run training
        final_step = self.training_loop.run(
            tokenizer=self.tokenizer,
            start_step=global_step,
            start_epoch=epoch,
            on_epoch_end=on_epoch_end,
        )

        return final_step


def main():
    """Command-line entry point for training."""
    parser = argparse.ArgumentParser(description="Run the main training loop for an experiment.")
    parser.add_argument("config_path", type=str, help="Path to the experiment's .yaml config file.")
    args = parser.parse_args()

    base_dir = find_project_root(__file__)
    abs_config_path = args.config_path if os.path.isabs(args.config_path) else os.path.join(base_dir, args.config_path)

    with open(abs_config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    try:
        config = ExperimentConfig(**config_data)
    except Exception as e:
        print(f"FATAL: Error validating configuration file '{abs_config_path}':\n{e}")
        return

    trainer = Trainer(config, base_dir)
    trainer.train()


if __name__ == '__main__':
    main()
