#!/usr/bin/env python3
"""
Experiment Workflow Runner

This script runs the complete experiment workflow from tokenization to training.
It handles all the steps needed to prepare and run an experiment on the remote server.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Optional, List
import yaml
import time

# Add the model_foundry package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_foundry.config import ExperimentConfig


def run_command(cmd: List[str], description: str, cwd: Optional[str] = None) -> bool:
    """
    Run a command with proper error handling and logging.
    
    Args:
        cmd: Command to run
        description: Description of what the command does
        cwd: Working directory for the command
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n=== {description} ===")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError as e:
        print(f"✗ Command not found: {e}")
        return False


def validate_config(config_path: str) -> Optional[ExperimentConfig]:
    """
    Validate and load the experiment configuration.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        ExperimentConfig if valid, None otherwise
    """
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        config = ExperimentConfig(**config_data)
        print(f"✓ Configuration loaded: {config.experiment_name}")
        return config
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return None


def check_data_exists(config: ExperimentConfig, base_dir: str) -> bool:
    """
    Check if the required data files exist.
    
    Args:
        config: Experiment configuration
        base_dir: Project base directory
        
    Returns:
        True if data exists, False otherwise
    """
    data_path = Path(base_dir) / config.data.training_corpus
    if not data_path.exists():
        print(f"✗ Training data not found: {data_path}")
        return False
    
    print(f"✓ Training data found: {data_path}")
    return True


def run_tokenization(config_path: str, base_dir: str) -> bool:
    """
    Run the tokenization pipeline.
    
    Args:
        config_path: Path to the configuration file
        base_dir: Project base directory
        
    Returns:
        True if successful, False otherwise
    """
    # Train tokenizer
    train_cmd = [
        "python", "-m", "model_foundry.tokenizer.train_tokenizer",
        "--config", config_path,
        "--base_dir", base_dir
    ]
    
    if not run_command(train_cmd, "Training tokenizer", base_dir):
        return False
    
    # Tokenize dataset
    tokenize_cmd = [
        "python", "-m", "model_foundry.tokenizer.tokenize_dataset",
        "--config", config_path,
        "--base_dir", base_dir
    ]
    
    if not run_command(tokenize_cmd, "Tokenizing dataset", base_dir):
        return False
    
    return True


def run_preprocessing(config_path: str, base_dir: str) -> bool:
    """
    Run the data preprocessing pipeline.
    
    Args:
        config_path: Path to the configuration file
        base_dir: Project base directory
        
    Returns:
        True if successful, False otherwise
    """
    preprocess_cmd = [
        "python", "-m", "model_foundry.cli", "preprocess",
        config_path
    ]
    
    return run_command(preprocess_cmd, "Running preprocessing pipeline", base_dir)


def generate_checkpoint_schedule(config_path: str, base_dir: str) -> bool:
    """
    Generate checkpoint schedule for the experiment.
    
    Args:
        config_path: Path to the configuration file
        base_dir: Project base directory
        
    Returns:
        True if successful, False otherwise
    """
    schedule_cmd = [
        "python", "scripts/generate_checkpoint_schedule.py",
        config_path
    ]
    
    return run_command(schedule_cmd, "Generating checkpoint schedule", base_dir)


def run_training(config_path: str, base_dir: str) -> bool:
    """
    Run the training pipeline.
    
    Args:
        config_path: Path to the configuration file
        base_dir: Project base directory
        
    Returns:
        True if successful, False otherwise
    """
    train_cmd = [
        "python", "-m", "model_foundry.trainer",
        config_path
    ]
    
    return run_command(train_cmd, "Running training", base_dir)


def main():
    """
    Main workflow runner.
    """
    parser = argparse.ArgumentParser(
        description="Run complete experiment workflow from tokenization to training"
    )
    parser.add_argument(
        "config_path",
        help="Path to the experiment configuration file"
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Project base directory (default: current directory)"
    )
    parser.add_argument(
        "--skip-tokenization",
        action="store_true",
        help="Skip tokenization step (if already done)"
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip preprocessing step (if already done)"
    )
    parser.add_argument(
        "--skip-schedule",
        action="store_true",
        help="Skip checkpoint schedule generation (if already done)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training step (for testing other steps)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    base_dir = Path(args.base_dir).resolve()
    config_path = args.config_path if Path(args.config_path).is_absolute() else base_dir / args.config_path
    
    print(f"=== Experiment Workflow Runner ===")
    print(f"Base directory: {base_dir}")
    print(f"Config file: {config_path}")
    print(f"Dry run: {args.dry_run}")
    
    if args.dry_run:
        print("\n=== DRY RUN MODE ===")
        print("The following steps would be executed:")
        if not args.skip_tokenization:
            print("  1. Train tokenizer")
            print("  2. Tokenize dataset")
        if not args.skip_preprocessing:
            print("  3. Run preprocessing pipeline")
        if not args.skip_schedule:
            print("  4. Generate checkpoint schedule")
        if not args.skip_training:
            print("  5. Run training")
        return
    
    # Validate configuration
    config = validate_config(str(config_path))
    if config is None:
        sys.exit(1)
    
    # Check data exists
    if not check_data_exists(config, str(base_dir)):
        sys.exit(1)
    
    start_time = time.time()
    
    # Step 1: Tokenization
    if not args.skip_tokenization:
        if not run_tokenization(str(config_path), str(base_dir)):
            print("✗ Tokenization failed. Stopping workflow.")
            sys.exit(1)
    else:
        print("\n=== Skipping tokenization ===")
    
    # Step 2: Preprocessing (if needed)
    if not args.skip_preprocessing and config.dataset_manipulation:
        if not run_preprocessing(str(config_path), str(base_dir)):
            print("✗ Preprocessing failed. Stopping workflow.")
            sys.exit(1)
    else:
        print("\n=== Skipping preprocessing (no manipulations or skipped) ===")
    
    # Step 3: Generate checkpoint schedule
    if not args.skip_schedule:
        if not generate_checkpoint_schedule(str(config_path), str(base_dir)):
            print("✗ Checkpoint schedule generation failed. Stopping workflow.")
            sys.exit(1)
    else:
        print("\n=== Skipping checkpoint schedule generation ===")
    
    # Step 4: Training
    if not args.skip_training:
        if not run_training(str(config_path), str(base_dir)):
            print("✗ Training failed.")
            sys.exit(1)
    else:
        print("\n=== Skipping training ===")
    
    elapsed_time = time.time() - start_time
    print(f"\n=== Workflow Complete ===")
    print(f"Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"Experiment: {config.experiment_name}")
    print("✓ All steps completed successfully!")


if __name__ == "__main__":
    main() 