#!/usr/bin/env python3
"""
Command Line Interface for the Model Foundry experimental framework.

This module provides a unified CLI for orchestrating the entire experimental pipeline,
from data preprocessing to model training and evaluation.
"""

import os
import sys
import yaml
import subprocess
import logging
from pathlib import Path
from typing import Optional, List
import typer
from typer import Option

# Add the model_foundry package to the path
sys.path.insert(0, str(Path(__file__).parent))

from .config import ExperimentConfig
from .trainer import Trainer
from .utils import find_project_root, set_seed
from .logging_utils import setup_logging

app = typer.Typer(
    name="model-foundry",
    help="Experimental framework for controlled rearing studies of language models",
    add_completion=False
)


@app.callback()
def main(
    log_dir: str = typer.Option("logs", help="Root directory for all log files"),
    log_level: str = typer.Option("INFO", help="Logging level"),
    experiment_name: str = typer.Option("default", help="Experiment identifier"),
):
    """
    This runs before any sub-command (train, preprocess, …).
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    # Use the root logger name so child loggers inherit handlers
    setup_logging("subject_drop_rearing", experiment=experiment_name,
                  log_dir=log_dir, level=level)
    
    # Store these in a way that subcommands can access them
    import os
    os.environ['CLI_LOG_LEVEL'] = log_level
    os.environ['CLI_EXPERIMENT_NAME'] = experiment_name


def load_config(config_path: str) -> ExperimentConfig:
    """Load and validate a configuration file."""
    base_dir = find_project_root(__file__)
    abs_config_path = config_path if os.path.isabs(config_path) else os.path.join(base_dir, config_path)
    
    if not os.path.exists(abs_config_path):
        raise typer.BadParameter(f"Configuration file not found: {abs_config_path}")
    
    with open(abs_config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    try:
        config = ExperimentConfig(**config_data)
        return config
    except Exception as e:
        raise typer.BadParameter(f"Error validating configuration file '{abs_config_path}': {e}")


@app.command()
def preprocess(
    config_path: str = typer.Argument(..., help="Path to the experiment's .yaml configuration file"),
    dry_run: bool = Option(False, "--dry-run", help="Show what would be executed without running it")
):
    """
    Run the dataset manipulation pipeline defined in the configuration.
    
    This command executes the preprocessing steps defined in the 'dataset_manipulation'
    section of the config file, calling the appropriate scripts from the preprocessing/
    directory.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"--- Preprocessing Pipeline: {config_path} ---")
    
    config = load_config(config_path)
    base_dir = find_project_root(__file__)
    
    # Check if dataset_manipulation is defined
    if not hasattr(config, 'dataset_manipulation') or not config.dataset_manipulation:
        logger.info("  - No dataset manipulation pipeline defined. Skipping preprocessing.")
        return
    
    logger.info(f"  - Found {len(config.dataset_manipulation)} preprocessing steps")
    
    # Execute each preprocessing step
    for i, step in enumerate(config.dataset_manipulation):
        step_type = step.get('type')
        input_path = step.get('input_path')
        output_path = step.get('output_path')
        
        if not all([step_type, input_path, output_path]):
            raise typer.BadParameter(f"Invalid preprocessing step {i}: missing required fields")
        
        # Resolve paths relative to project root
        abs_input_path = input_path if os.path.isabs(input_path) else os.path.join(base_dir, input_path)
        abs_output_path = output_path if os.path.isabs(output_path) else os.path.join(base_dir, output_path)
        
        # Find the preprocessing script
        script_path = os.path.join(base_dir, "preprocessing", f"{step_type}.py")
        if not os.path.exists(script_path):
            raise typer.BadParameter(f"Preprocessing script not found: {script_path}")
        
        logger.info(f"  - Step {i+1}: {step_type}")
        logger.info(f"    Input:  {abs_input_path}")
        logger.info(f"    Output: {abs_output_path}")
        
        if dry_run:
            logger.info(f"    [DRY RUN] Would execute: python {script_path} --input_dir {abs_input_path} --output_dir {abs_output_path}")
            continue
        
        # Execute the preprocessing script
        cmd = [
            sys.executable, script_path,
            "--input_dir", abs_input_path,
            "--output_dir", abs_output_path
        ]
        
        # Add verbose flag if CLI is running with verbose logging
        cli_log_level = os.environ.get('CLI_LOG_LEVEL', 'INFO')
        if cli_log_level.upper() in ["DEBUG", "VERBOSE"]:
            cmd.append("--verbose")
        
        # Add additional parameters if specified
        if 'parameters' in step:
            for key, value in step['parameters'].items():
                if isinstance(value, bool):
                    # For boolean flags, only add the flag if True
                    if value:
                        cmd.extend([f"--{key}"])
                else:
                    # For non-boolean values, add both flag and value
                    cmd.extend([f"--{key}", str(value)])
        
        logger.info(f"    Executing: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            logger.info(f"    ✓ Completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"    ✗ Failed with exit code {e.returncode}")
            raise typer.Exit(1)
    
    logger.info("  - Preprocessing pipeline completed successfully")


@app.command()
def train_tokenizer(
    config_path: str = typer.Argument(..., help="Path to the experiment's .yaml configuration file")
):
    """
    Train a SentencePiece tokenizer for the experiment.
    
    This command trains a new tokenizer on the training corpus specified in the config,
    using the parameters defined in the 'tokenizer' section.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"--- Training Tokenizer: {config_path} ---")
    
    config = load_config(config_path)
    base_dir = find_project_root(__file__)
    
    # Import and run the tokenizer training
    from .tokenizer.train_tokenizer import train_tokenizer_from_config
    
    abs_config_path = config_path if os.path.isabs(config_path) else os.path.join(base_dir, config_path)
    train_tokenizer_from_config(abs_config_path)


@app.command()
def tokenize_dataset(
    config_path: str = typer.Argument(..., help="Path to the experiment's .yaml configuration file")
):
    """
    Tokenize the training dataset using the experiment's tokenizer.
    
    This command loads the training corpus, tokenizes it using the trained SentencePiece
    model, and saves the tokenized dataset to disk for training.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"--- Tokenizing Dataset: {config_path} ---")
    
    config = load_config(config_path)
    base_dir = find_project_root(__file__)
    
    # Import and run the dataset tokenization
    from .tokenizer.tokenize_dataset import tokenize_dataset_from_config
    
    abs_config_path = config_path if os.path.isabs(config_path) else os.path.join(base_dir, config_path)
    tokenize_dataset_from_config(abs_config_path)


@app.command()
def preprocess_data(
    config_path: str = typer.Argument(..., help="Path to the experiment's .yaml configuration file"),
    force_reprocess: bool = Option(False, "--force", help="Force reprocessing even if chunked data exists")
):
    """
    Preprocess the tokenized dataset into fixed-length chunks.
    
    This command loads the tokenized dataset, creates fixed-length chunks,
    and saves the processed dataset to disk for efficient training.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"--- Data Preprocessing: {config_path} ---")
    
    config = load_config(config_path)
    base_dir = find_project_root(__file__)
    
    # Import and run the data preprocessing
    from .data import create_data_processor
    
    data_processor = create_data_processor(config, base_dir)
    success = data_processor.preprocess_data(force_reprocess=force_reprocess)
    
    if not success:
        raise typer.Exit(1)


@app.command()
def generate_checkpoints(
    config_path: str = typer.Argument(..., help="Path to the experiment's .yaml configuration file"),
    output_path: Optional[str] = Option(None, "--output", "-o", help="Output path for updated config (default: overwrite original)"),
    first_epoch_checkpoints: int = Option(20, "--first-epoch", help="Number of checkpoints in the first epoch"),
    subsequent_spacing: str = Option("log", "--spacing", help="Spacing for subsequent epochs: 'linear' or 'log'"),
    log_base: int = Option(2, "--log-base", help="Base for logarithmic spacing (default: 2)"),
    linear_interval: Optional[int] = Option(None, "--linear-interval", help="Steps between checkpoints for linear spacing"),
    min_interval: int = Option(100, "--min-interval", help="Minimum interval between checkpoints"),
    min_checkpoints_per_epoch: int = Option(5, "--min-per-epoch", help="Minimum checkpoints per epoch (after first epoch)"),
):
    """
    Generate an optimal checkpoint schedule for a training experiment.
    
    This command analyzes the experiment configuration and generates a checkpoint
    schedule that balances checkpoint frequency with storage efficiency.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"--- Checkpoint Schedule Generation: {config_path} ---")
    
    # Import the checkpoint generation functions directly
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from generate_checkpoint_schedule import (
        generate_checkpoint_schedule, 
        CheckpointGenerationConfig,
        save_schedule_to_config
    )
    from model_foundry.config import ExperimentConfig
    
    # Load configuration
    base_dir = find_project_root(__file__)
    abs_config_path = config_path if Path(config_path).is_absolute() else Path(base_dir) / config_path
    
    if not abs_config_path.exists():
        logger.error(f"Error: Configuration file not found: {abs_config_path}")
        raise typer.Exit(1)
    
    # Load experiment config
    with open(abs_config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    try:
        config = ExperimentConfig(**config_data)
    except Exception as e:
        logger.error(f"Error: Invalid configuration file: {e}")
        raise typer.Exit(1)
    
    # Create generation config
    generation_config = CheckpointGenerationConfig(
        first_epoch_checkpoints=first_epoch_checkpoints,
        subsequent_epochs_spacing=subsequent_spacing,
        log_base=log_base,
        linear_interval=linear_interval,
        min_interval=min_interval,
        min_checkpoints_per_epoch=min_checkpoints_per_epoch
    )
    
    # Generate schedule
    schedule = generate_checkpoint_schedule(config, str(base_dir), generation_config)
    
    # Save to config file
    save_schedule_to_config(schedule, str(abs_config_path), output_path)
    
    logger.info(f"\n✓ Checkpoint schedule generation complete!")


@app.command()
def run(
    config_path: str = typer.Argument(..., help="Path to the experiment's .yaml configuration file"),
    resume: bool = Option(False, "--resume", help="Resume training from the latest checkpoint")
):
    """
    Run the main training loop for the experiment.
    
    This command loads the configuration, prepares the data and model, and executes
    the training loop with the specified hyperparameters.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"--- Running Training: {config_path} ---")
    
    config = load_config(config_path)
    base_dir = find_project_root(__file__)
    
    # Set the resume flag if requested
    if resume:
        config.training.resume_from_checkpoint = True
    
    # Create and run the trainer with proper cleanup
    trainer = None
    try:
        trainer = Trainer(config, base_dir)
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if trainer and hasattr(trainer, '_cleanup_gpu'):
            trainer._cleanup_gpu()
        raise typer.Exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if trainer and hasattr(trainer, '_cleanup_gpu'):
            trainer._cleanup_gpu()
        raise typer.Exit(1)


@app.command()
def validate_config(
    config_path: str = typer.Argument(..., help="Path to the experiment's .yaml configuration file")
):
    """
    Validate a configuration file without running any commands.
    
    This command checks that the configuration file is valid and all required
    paths and parameters are properly defined.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"--- Validating Configuration: {config_path} ---")
    
    try:
        config = load_config(config_path)
        logger.info("  ✓ Configuration is valid")
        logger.info(f"  - Experiment name: {config.experiment_name}")
        logger.info(f"  - Training steps: {config.training.train_steps}")
        logger.info(f"  - Model layers: {config.model.layers}")
        logger.info(f"  - Vocab size: {config.tokenizer.vocab_size}")
        
        # Check if required files exist
        base_dir = find_project_root(__file__)
        
        # Check training corpus
        corpus_path = config.data.training_corpus
        abs_corpus_path = corpus_path if os.path.isabs(corpus_path) else os.path.join(base_dir, corpus_path)
        if os.path.exists(abs_corpus_path):
            logger.info(f"  ✓ Training corpus exists: {abs_corpus_path}")
        else:
            logger.warning(f"  ⚠ Training corpus not found: {abs_corpus_path}")
        
        # Check tokenizer directory
        tokenizer_dir = config.tokenizer.output_dir
        abs_tokenizer_dir = tokenizer_dir if os.path.isabs(tokenizer_dir) else os.path.join(base_dir, tokenizer_dir)
        if os.path.exists(abs_tokenizer_dir):
            logger.info(f"  ✓ Tokenizer directory exists: {abs_tokenizer_dir}")
        else:
            logger.warning(f"  ⚠ Tokenizer directory not found: {abs_tokenizer_dir}")
        
    except Exception as e:
        logger.error(f"  ✗ Configuration validation failed: {e}")
        raise typer.Exit(1)


@app.command()
def info():
    """
    Display information about the model foundry framework.
    """
    print("Model Foundry - Controlled Rearing Study Framework")
    print("=" * 50)
    print()
    print("This framework enables reproducible experiments for investigating")
    print("grammatical rule acquisition in language models through controlled")
    print("dataset manipulations and systematic evaluation.")
    print()
    print("Available Commands:")
    print("  preprocess      - Run dataset manipulation pipeline")
    print("  train-tokenizer - Train SentencePiece tokenizer")
    print("  tokenize-dataset- Tokenize training data")
    print("  run            - Execute training loop")
    print("  validate-config - Validate configuration file")
    print()
    print("For detailed help on any command, use: model-foundry <command> --help")


if __name__ == "__main__":
    app() 