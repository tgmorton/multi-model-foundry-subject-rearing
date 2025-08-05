#!/usr/bin/env python3
"""
Checkpoint Schedule Generator for Model Foundry

This script generates optimal checkpoint schedules based on dataset characteristics
and training parameters. It implements intelligent scheduling that adapts to
different dataset sizes and training configurations.
"""

import math
import yaml
from pathlib import Path
from typing import Dict, List, Set, Optional
import typer
from pydantic import BaseModel, field_validator

# Add the model_foundry package to the path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_foundry.config import ExperimentConfig
from model_foundry.data import create_data_processor


class CheckpointGenerationConfig(BaseModel):
    """Configuration for checkpoint schedule generation."""
    
    # First epoch configuration
    first_epoch_checkpoints: int = 20
    
    # Subsequent epochs configuration
    subsequent_epochs_spacing: str = "log"  # "linear" or "log"
    log_base: int = 2
    linear_interval: Optional[int] = None
    
    # Minimum interval between checkpoints
    min_interval: int = 100
    
    # Minimum number of checkpoints per epoch (for epochs after the first)
    min_checkpoints_per_epoch: int = 5
    
    @field_validator("subsequent_epochs_spacing")
    @classmethod
    def validate_spacing(cls, v: str) -> str:
        """Validate spacing type."""
        if v not in ["linear", "log"]:
            raise ValueError("subsequent_epochs_spacing must be 'linear' or 'log'")
        return v


def estimate_dataset_size(config: ExperimentConfig, base_dir: str) -> str:
    """
    Estimate the dataset size category based on the experiment configuration.
    
    Args:
        config: Experiment configuration
        base_dir: Project base directory
        
    Returns:
        Dataset size category ('small', 'medium', 'large', 'xlarge')
    """
    try:
        # Try to load the data processor to get actual dataset stats
        data_processor = create_data_processor(config, base_dir)
        
        # Check if chunked data exists
        if data_processor._load_chunked_dataset() is not None:
            dataset = data_processor._load_chunked_dataset()
            num_chunks = len(dataset)
            total_tokens = num_chunks * config.data.max_sequence_length
            
            # Categorize based on total tokens
            if total_tokens < 10_000_000:
                return "small"
            elif total_tokens < 25_000_000:
                return "medium"
            elif total_tokens < 50_000_000:
                return "large"
            else:
                return "xlarge"
        else:
            # Fallback to estimation based on corpus path
            corpus_path = config.data.training_corpus
            if "10M" in corpus_path or "small" in corpus_path:
                return "small"
            elif "25M" in corpus_path or "medium" in corpus_path:
                return "medium"
            elif "50M" in corpus_path or "large" in corpus_path:
                return "large"
            elif "100M" in corpus_path or "xlarge" in corpus_path:
                return "xlarge"
            else:
                return "medium"  # Default
                
    except Exception as e:
        print(f"Warning: Could not estimate dataset size: {e}")
        return "medium"  # Default fallback


def calculate_steps_per_epoch(config: ExperimentConfig, base_dir: str) -> int:
    """
    Calculate the actual number of steps per epoch based on dataset size and batch configuration.
    
    Args:
        config: Experiment configuration
        base_dir: Project base directory
        
    Returns:
        Number of steps per epoch
    """
    try:
        # Try to get actual dataset size from data processor
        data_processor = create_data_processor(config, base_dir)
        dataset = data_processor._load_chunked_dataset()
        
        if dataset is not None:
            # Calculate based on actual dataset size
            num_chunks = len(dataset)
            effective_batch_size = config.data.batch_size * config.training.gradient_accumulation_steps
            
            steps_per_epoch = math.ceil(num_chunks / effective_batch_size)
            print(f"  - Actual dataset chunks: {num_chunks:,}")
            print(f"  - Effective batch size: {effective_batch_size}")
            print(f"  - Calculated steps per epoch: {steps_per_epoch:,}")
            return steps_per_epoch
        else:
            # Fallback to estimation - use a reasonable default if train_steps is None
            print(f"  - Warning: Using estimated steps per epoch")
            if config.training.train_steps is not None:
                return config.training.train_steps // config.training.epochs
            else:
                # Conservative estimate when we have no data
                print(f"  - Warning: No dataset or train_steps available, using conservative estimate")
                return 100  # Conservative fallback
            
    except Exception as e:
        print(f"  - Warning: Could not calculate actual steps per epoch: {e}")
        if config.training.train_steps is not None:
            return config.training.train_steps // config.training.epochs
        else:
            # Conservative estimate when we have no data
            print(f"  - Warning: No dataset or train_steps available, using conservative estimate")
            return 100  # Conservative fallback


def generate_first_epoch_checkpoints(steps_per_epoch: int, num_checkpoints: int) -> List[int]:
    """
    Generate checkpoints for the first epoch with specified number of checkpoints.
    
    Args:
        steps_per_epoch: Number of steps in the first epoch
        num_checkpoints: Number of checkpoints to generate
        
    Returns:
        List of checkpoint steps for the first epoch
    """
    if num_checkpoints <= 0:
        return []
    
    if num_checkpoints == 1:
        return [steps_per_epoch]
    
    # Distribute checkpoints evenly across the first epoch
    interval = steps_per_epoch / (num_checkpoints - 1)
    checkpoints = []
    
    for i in range(num_checkpoints):
        step = round(i * interval)
        if step <= steps_per_epoch:
            checkpoints.append(step)
    
    return checkpoints


def generate_subsequent_epoch_checkpoints(
    epoch_start: int,
    epoch_end: int,
    spacing: str,
    log_base: int = 2,
    linear_interval: Optional[int] = None,
    min_checkpoints_per_epoch: int = 5
) -> List[int]:
    """
    Generate checkpoints for subsequent epochs with specified spacing.
    
    Args:
        epoch_start: Starting step of the epoch
        epoch_end: Ending step of the epoch
        spacing: "linear" or "log"
        log_base: Base for logarithmic spacing
        linear_interval: Steps between checkpoints for linear spacing
        min_checkpoints_per_epoch: Minimum number of checkpoints per epoch
        
    Returns:
        List of checkpoint steps for the epoch
    """
    checkpoints = []
    epoch_length = epoch_end - epoch_start
    
    if spacing == "linear":
        if linear_interval is None:
            # Calculate reasonable interval based on epoch length
            linear_interval = max(1, epoch_length // 10)
        
        current_step = epoch_start + linear_interval
        while current_step < epoch_end:
            checkpoints.append(current_step)
            current_step += linear_interval
    
    elif spacing == "log":
        current_step = epoch_start + log_base
        while current_step < epoch_end:
            checkpoints.append(current_step)
            current_step *= log_base
    
    # Ensure minimum number of checkpoints per epoch
    if len(checkpoints) < min_checkpoints_per_epoch and epoch_length > min_checkpoints_per_epoch:
        # Calculate evenly spaced checkpoints to meet minimum requirement
        interval = epoch_length // (min_checkpoints_per_epoch + 1)  # +1 to avoid boundary
        checkpoints = []
        
        for i in range(1, min_checkpoints_per_epoch + 1):
            checkpoint_step = epoch_start + (i * interval)
            if checkpoint_step < epoch_end:
                checkpoints.append(checkpoint_step)
        
        # Sort and remove duplicates
        checkpoints = sorted(list(set(checkpoints)))
    
    return checkpoints


def generate_checkpoint_schedule(
    config: ExperimentConfig,
    base_dir: str,
    generation_config: CheckpointGenerationConfig
) -> List[int]:
    """
    Generate a checkpoint schedule based on the new requirements.
    
    Args:
        config: Experiment configuration
        base_dir: Project base directory
        generation_config: Checkpoint generation configuration
        
    Returns:
        List of training steps at which to save checkpoints
    """
    print(f"--- Generating Checkpoint Schedule for: {config.experiment_name} ---")
    
    # Calculate actual steps per epoch
    steps_per_epoch = calculate_steps_per_epoch(config, base_dir)
    
    # Calculate total steps - handle null values with automatic calculation
    if config.training.train_steps is None:
        total_steps = steps_per_epoch * config.training.epochs
        print(f"  - Auto-calculated total training steps: {total_steps:,} ({config.training.epochs} epochs × {steps_per_epoch} steps/epoch)")
    else:
        total_steps = config.training.train_steps
        
    num_epochs = config.training.epochs
    
    print(f"  - Total training steps: {total_steps:,}")
    print(f"  - Steps per epoch: {steps_per_epoch:,}")
    print(f"  - Training epochs: {num_epochs}")
    print(f"  - First epoch checkpoints: {generation_config.first_epoch_checkpoints}")
    print(f"  - Subsequent epochs spacing: {generation_config.subsequent_epochs_spacing}")
    
    checkpoint_steps: Set[int] = set()
    
    # Generate first epoch checkpoints
    first_epoch_checkpoints = generate_first_epoch_checkpoints(
        steps_per_epoch, 
        generation_config.first_epoch_checkpoints
    )
    checkpoint_steps.update(first_epoch_checkpoints)
    print(f"  - First epoch checkpoints: {len(first_epoch_checkpoints)}")
    
    # Generate checkpoints for subsequent epochs
    for epoch in range(2, num_epochs + 1):
        epoch_start = (epoch - 1) * steps_per_epoch
        epoch_end = epoch * steps_per_epoch
        
        if epoch_end > total_steps:
            epoch_end = total_steps
        
        epoch_checkpoints = generate_subsequent_epoch_checkpoints(
            epoch_start,
            epoch_end,
            generation_config.subsequent_epochs_spacing,
            generation_config.log_base,
            generation_config.linear_interval,
            generation_config.min_checkpoints_per_epoch
        )
        
        # Always add epoch boundary checkpoint
        epoch_checkpoints.append(epoch_end)
        
        checkpoint_steps.update(epoch_checkpoints)
        print(f"  - Epoch {epoch} checkpoints: {len(epoch_checkpoints)}")
    
    # Add final step if not already included
    checkpoint_steps.add(total_steps)
    
    # Convert to sorted list
    final_schedule = sorted(list(checkpoint_steps))
    
    print(f"  - Generated {len(final_schedule)} total checkpoint steps")
    print(f"  - Schedule: {final_schedule[:10]}{'...' if len(final_schedule) > 10 else ''}")
    
    return final_schedule


def save_schedule_to_config(
    schedule: List[int],
    config_path: str,
    output_path: Optional[str] = None
) -> None:
    """
    Save the checkpoint schedule to a configuration file.
    
    Args:
        schedule: List of checkpoint steps
        config_path: Path to the original config file
        output_path: Path to save the updated config (if None, overwrites original)
    """
    if output_path is None:
        output_path = config_path
    
    # Load original config
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Update with checkpoint schedule
    if 'training' not in config_data:
        config_data['training'] = {}
    
    config_data['training']['checkpoint_schedule'] = schedule
    
    # Save updated config
    with open(output_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    print(f"  ✓ Saved checkpoint schedule to: {output_path}")


def main(
    config_path: str = typer.Argument(..., help="Path to the experiment's .yaml configuration file"),
    output_path: Optional[str] = typer.Option(None, "--output", "-o", help="Output path for updated config (default: overwrite original)"),
    first_epoch_checkpoints: int = typer.Option(20, "--first-epoch", help="Number of checkpoints in the first epoch"),
    subsequent_spacing: str = typer.Option("log", "--spacing", help="Spacing for subsequent epochs: 'linear' or 'log'"),
    log_base: int = typer.Option(2, "--log-base", help="Base for logarithmic spacing (default: 2)"),
    linear_interval: Optional[int] = typer.Option(None, "--linear-interval", help="Steps between checkpoints for linear spacing"),
    min_interval: int = typer.Option(100, "--min-interval", help="Minimum interval between checkpoints"),
    min_checkpoints_per_epoch: int = typer.Option(5, "--min-per-epoch", help="Minimum checkpoints per epoch (after first epoch)"),
):
    """
    Generate an optimal checkpoint schedule for a training experiment.
    
    This script analyzes the experiment configuration and generates a checkpoint
    schedule that balances checkpoint frequency with storage efficiency.
    """
    # Load configuration
    base_dir = Path(__file__).parent.parent
    abs_config_path = config_path if Path(config_path).is_absolute() else base_dir / config_path
    
    if not abs_config_path.exists():
        print(f"Error: Configuration file not found: {abs_config_path}")
        raise typer.Exit(1)
    
    # Load experiment config
    with open(abs_config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    try:
        config = ExperimentConfig(**config_data)
    except Exception as e:
        print(f"Error: Invalid configuration file: {e}")
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
    
    print(f"\n✓ Checkpoint schedule generation complete!")


if __name__ == "__main__":
    typer.run(main) 