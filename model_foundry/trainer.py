"""
Main Trainer class for the Model Foundry framework.

This module coordinates the training process by orchestrating the model,
data processing, checkpointing, and training loop components.
"""

import argparse
import os
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

    def _setup_memory_management(self):
        """Configure CUDA memory management settings."""
        if not torch.cuda.is_available():
            return

        # Set memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.95)

        # Configure allocator for better performance
        if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ or not os.environ['PYTORCH_CUDA_ALLOC_CONF']:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

        # Enable cudnn benchmarking
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        print("  - CUDA memory management configured (95% memory limit, max_split_size_mb:512)")

    def _calculate_training_parameters(self):
        """Calculate training parameters based on dataset size if not explicitly set."""
        try:
            steps_per_epoch = self.data_processor.get_training_steps_per_epoch()
        except:
            print("  ⚠️  Could not determine exact dataset size, using estimates")
            steps_per_epoch = 100

        # Calculate train_steps if not specified
        if self.config.training.train_steps is None:
            calculated_train_steps = steps_per_epoch * self.config.training.epochs
            print(f"  - Auto-calculated train_steps: {calculated_train_steps} "
                  f"({self.config.training.epochs} epochs × {steps_per_epoch} steps/epoch)")
            self.config.training.train_steps = calculated_train_steps

        # Calculate warmup_steps if not specified
        if self.config.training.warmup_steps is None:
            calculated_warmup_steps = int(self.config.training.train_steps * self.config.training.warmup_ratio)
            print(f"  - Auto-calculated warmup_steps: {calculated_warmup_steps} "
                  f"({self.config.training.warmup_ratio:.1%} of total steps)")
            self.config.training.warmup_steps = calculated_warmup_steps

        print(f"  - Final training configuration:")
        print(f"    - Epochs: {self.config.training.epochs}")
        print(f"    - Steps per epoch: {steps_per_epoch}")
        print(f"    - Total training steps: {self.config.training.train_steps}")
        print(f"    - Warmup steps: {self.config.training.warmup_steps}")

    def _prepare_data(self):
        """Load and prepare the dataset and dataloader."""
        if not self.data_processor.preprocess_data():
            raise RuntimeError("Data preprocessing failed")

        self.dataloader = self.data_processor.create_dataloader(self.tokenizer)

    def _initialize_model(self):
        """Initialize the model with appropriate optimizations."""
        # Disable torch compile globally to avoid ldconfig issues
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.disable()

        try:
            # Try Flash Attention 2 first
            print("  - Attempting to create model with Flash Attention 2...")
            self.model = create_model(
                self.config,
                attn_implementation="flash_attention_2"
            ).to(self.device)
            print("  - Successfully initialized model with Flash Attention 2")
        except (ImportError, ValueError) as e:
            # Fall back to standard attention
            print(f"  - Flash Attention 2 not available ({e}), falling back to standard attention")
            self.model = create_model(self.config).to(self.device)
            print("  - Model created with standard attention")

        # Apply torch.compile if configured
        self._apply_torch_compile()

        # Apply memory optimizations
        self._apply_memory_optimizations()

    def _apply_torch_compile(self):
        """Apply torch.compile optimization if configured."""
        compile_mode = getattr(self.config.training, 'compile_mode', None)

        if not compile_mode or str(compile_mode).lower() == 'none':
            print(f"  - Skipping torch.compile (compile_mode='{compile_mode}')")
            return

        try:
            print(f"  - Compiling model with torch.compile(mode='{compile_mode}')...")
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True

            if compile_mode == 'reduce-overhead':
                self.model = torch.compile(self.model, mode="reduce-overhead", backend="eager")
            elif compile_mode == 'max-autotune':
                self.model = torch.compile(self.model, mode="max-autotune", backend="eager")
            else:
                self.model = torch.compile(self.model, mode="default", backend="eager")

            print("  - Model compilation successful")
        except Exception as e:
            print(f"  - Warning: torch.compile failed ({e}), continuing without compilation")

    def _apply_memory_optimizations(self):
        """Apply memory optimization settings."""
        if self.config.training.use_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("  - TF32 enabled for faster training on Ampere+ GPUs")

        if self.config.training.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("  - Gradient checkpointing enabled to save memory")

    def _initialize_optimizer_and_scheduler(self):
        """Initialize optimizer and learning rate scheduler."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
            eps=self.config.training.adam_epsilon
        )

        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=self.config.training.train_steps
        )

    def _load_tokenizer(self):
        """Load the tokenizer from its directory."""
        tokenizer_path = os.path.join(self.base_dir, self.config.tokenizer.output_dir)
        self.tokenizer = load_tokenizer(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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

    def train(self):
        """Main training entry point."""
        logger = setup_logging(
            "trainer",
            experiment=self.config.experiment_name,
            log_dir=self.config.logging.dir,
            level=getattr(logging, self.config.logging.level)
        )
        logger.info(f"--- Starting Training Run for: {self.config.experiment_name} ---")

        try:
            return self._train_loop()
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

    def _train_loop(self):
        """Internal training loop implementation."""
        print("Starting training setup...")

        set_seed(self.config.random_seed)

        # Calculate training parameters
        print("  - Calculating training parameters from dataset...")
        self._calculate_training_parameters()

        # Initialize components
        print("  - Initializing model...")
        self._initialize_model()

        print("  - Initializing optimizer and scheduler...")
        self._initialize_optimizer_and_scheduler()

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            self.config, self.base_dir, self.git_commit_hash
        )

        # Try to load from checkpoint
        model, tokenizer, global_step, epoch = self.checkpoint_manager.load_checkpoint(
            model_factory=lambda: create_model(self.config),
            device=self.device,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler
        )

        if model is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            # Load tokenizer from directory
            print("  - Loading tokenizer...")
            self._load_tokenizer()

        # Prepare data
        print("  - Preparing data...")
        self._prepare_data()

        # Initialize W&B logging
        if self.config.logging.use_wandb:
            wandb.init(
                project=self.config.logging.wandb_project,
                name=self.config.experiment_name,
                config=self.config.model_dump(),
                resume="allow",
                id=wandb.util.generate_id()
            )

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
            start_epoch=epoch
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
