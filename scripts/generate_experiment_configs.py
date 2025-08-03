#!/usr/bin/env python3
"""
Generate Experiment Configurations

This script generates configuration files for all experiments based on the processed
data folders. It creates both tokenization and training configurations for each
experiment, ensuring proper data paths and experiment naming.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import typer

# Add the model_foundry package to the path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_foundry.config import ExperimentConfig


def get_processed_experiments(base_dir: str) -> List[Dict[str, str]]:
    """
    Scan the processed data directory to find all experiments.
    
    Args:
        base_dir: Project base directory
        
    Returns:
        List of experiment info dictionaries
    """
    processed_dir = Path(base_dir) / "data" / "processed"
    experiments = []
    
    if not processed_dir.exists():
        print(f"Warning: Processed data directory not found: {processed_dir}")
        return experiments
    
    for exp_dir in processed_dir.iterdir():
        if exp_dir.is_dir() and not exp_dir.name.startswith('.'):
            # Extract experiment number and name
            exp_name = exp_dir.name
            
            # Determine experiment number
            if exp_name == "train_remove_expletives_90M":
                exp_num = 0
                exp_display_name = "baseline"
            elif exp_name.startswith("exp"):
                # Extract number from exp1_remove_expletives format
                parts = exp_name.split('_')
                if parts[0].startswith('exp') and parts[0][3:].isdigit():
                    exp_num = int(parts[0][3:])
                    exp_display_name = '_'.join(parts[1:])
                else:
                    continue
            else:
                continue
            
            experiments.append({
                'number': exp_num,
                'name': exp_name,
                'display_name': exp_display_name,
                'path': str(exp_dir),
                'config_name': f"experiment_{exp_num}_{exp_display_name}"
            })
    
    return sorted(experiments, key=lambda x: x['number'])


def create_tokenizer_config(exp_info: Dict[str, str], base_dir: str) -> Dict:
    """
    Create tokenizer configuration for an experiment.
    
    Args:
        exp_info: Experiment information dictionary
        base_dir: Project base directory
        
    Returns:
        Tokenizer configuration dictionary
    """
    return {
        'output_dir': f"tokenizers/exp{exp_info['number']}_{exp_info['display_name']}/",
        'vocab_size': 50004
    }


def create_data_config(exp_info: Dict[str, str], base_dir: str) -> Dict:
    """
    Create data configuration for an experiment.
    
    Args:
        exp_info: Experiment information dictionary
        base_dir: Project base directory
        
    Returns:
        Data configuration dictionary
    """
    return {
        'source_corpus': exp_info['path'],
        'training_corpus': exp_info['path'],
        'batch_size': 256,
        'max_sequence_length': 128
    }


def create_model_config() -> Dict:
    """
    Create standard model configuration.
    
    Returns:
        Model configuration dictionary
    """
    return {
        'layers': 12,
        'embedding_size': 768,
        'hidden_size': 768,
        'intermediate_hidden_size': 3072,
        'attention_heads': 12,
        'activation_function': 'GELU',
        'dropout': 0.1,
        'attention_dropout': 0.1
    }


def create_training_config(exp_info: Dict[str, str]) -> Dict:
    """
    Create training configuration for an experiment.
    
    Args:
        exp_info: Experiment information dictionary
        
    Returns:
        Training configuration dictionary
    """
    return {
        'output_dir': f"models/exp{exp_info['number']}_{exp_info['display_name']}/",
        'learning_rate': 0.0001,
        'adam_beta1': 0.9,
        'adam_beta2': 0.999,
        'adam_epsilon': 1.0e-6,
        'warmup_steps': 10000,
        'train_steps': 1000000,
        'epochs': 20,
        'checkpointing_strategy': 'log_steps_first_epoch',
        'auto_generate_checkpoints': True,
        'first_epoch_checkpoints': 20,
        'subsequent_epochs_spacing': 'log',
        'log_base': 2,
        'min_checkpoint_interval': 100,
        'resume_from_checkpoint': False,
        'use_amp': True,  # Enable AMP by default
        'gradient_accumulation_steps': 1,
        'use_tf32': True,
        'use_gradient_checkpointing': False
    }


def create_logging_config() -> Dict:
    """
    Create standard logging configuration.
    
    Returns:
        Logging configuration dictionary
    """
    return {
        'level': 'INFO',
        'dir': 'logs',
        'use_wandb': True,
        'wandb_project': 'just-drop-the-subject'
    }


def create_experiment_config(exp_info: Dict[str, str], base_dir: str) -> Dict:
    """
    Create complete experiment configuration.
    
    Args:
        exp_info: Experiment information dictionary
        base_dir: Project base directory
        
    Returns:
        Complete experiment configuration dictionary
    """
    return {
        'experiment_name': f"exp{exp_info['number']}_{exp_info['display_name']}",
        'data': create_data_config(exp_info, base_dir),
        'tokenizer': create_tokenizer_config(exp_info, base_dir),
        'model': create_model_config(),
        'training': create_training_config(exp_info),
        'logging': create_logging_config(),
        'random_seed': 42,
        'dataset_manipulation': []  # Empty for processed data experiments
    }


def save_config(config: Dict, config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration
    """
    config_dir = Path(config_path).parent
    config_dir.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
    
    print(f"  ✓ Saved: {config_path}")


def create_tokenization_script(exp_info: Dict[str, str], base_dir: str) -> str:
    """
    Create a tokenization script for an experiment.
    
    Args:
        exp_info: Experiment information dictionary
        base_dir: Project base directory
        
    Returns:
        Script content as string
    """
    script_content = f"""#!/bin/bash
# Tokenization script for {exp_info['config_name']}

set -e

echo "=== Tokenizing {exp_info['display_name']} experiment ==="

# Set up paths
BASE_DIR="{base_dir}"
CONFIG_FILE="$BASE_DIR/configs/{exp_info['config_name']}.yaml"

# Create tokenizer directory
TOKENIZER_DIR="$BASE_DIR/tokenizers/exp{exp_info['number']}_{exp_info['display_name']}/"
mkdir -p "$TOKENIZER_DIR"

echo "Training tokenizer..."
python -m model_foundry.tokenizer.train_tokenizer \\
    --config "$CONFIG_FILE" \\
    --base_dir "$BASE_DIR"

echo "Tokenizing dataset..."
python -m model_foundry.tokenizer.tokenize_dataset \\
    --config "$CONFIG_FILE" \\
    --base_dir "$BASE_DIR"

echo "✓ Tokenization complete for {exp_info['display_name']}"
"""
    return script_content


def create_training_script(exp_info: Dict[str, str], base_dir: str) -> str:
    """
    Create a training script for an experiment.
    
    Args:
        exp_info: Experiment information dictionary
        base_dir: Project base directory
        
    Returns:
        Script content as string
    """
    script_content = f"""#!/bin/bash
# Training script for {exp_info['config_name']}

set -e

echo "=== Training {exp_info['display_name']} experiment ==="

# Set up paths
BASE_DIR="{base_dir}"
CONFIG_FILE="$BASE_DIR/configs/{exp_info['config_name']}.yaml"

# Generate checkpoint schedule
echo "Generating checkpoint schedule..."
python scripts/generate_checkpoint_schedule.py "$CONFIG_FILE"

# Run training
echo "Starting training..."
python -m model_foundry.trainer "$CONFIG_FILE"

echo "✓ Training complete for {exp_info['display_name']}"
"""
    return script_content


def main(
    base_dir: str = typer.Option(".", "--base-dir", help="Project base directory"),
    output_dir: str = typer.Option("configs", "--output-dir", help="Output directory for configs"),
    create_scripts: bool = typer.Option(True, "--create-scripts", help="Create tokenization and training scripts"),
    scripts_dir: str = typer.Option("scripts", "--scripts-dir", help="Directory for generated scripts")
):
    """
    Generate configuration files for all experiments based on processed data.
    """
    base_path = Path(base_dir).resolve()
    output_path = Path(output_dir)
    scripts_path = Path(scripts_dir)
    
    print(f"=== Generating Experiment Configurations ===")
    print(f"Base directory: {base_path}")
    print(f"Output directory: {output_path}")
    
    # Get all experiments
    experiments = get_processed_experiments(str(base_path))
    
    if not experiments:
        print("No experiments found in processed data directory.")
        return
    
    print(f"Found {len(experiments)} experiments:")
    for exp in experiments:
        print(f"  - {exp['config_name']}: {exp['display_name']}")
    
    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    if create_scripts:
        scripts_path.mkdir(parents=True, exist_ok=True)
    
    # Generate configurations for each experiment
    for exp_info in experiments:
        print(f"\n--- Generating config for {exp_info['config_name']} ---")
        
        # Create configuration
        config = create_experiment_config(exp_info, str(base_path))
        
        # Save configuration
        config_file = output_path / f"{exp_info['config_name']}.yaml"
        save_config(config, str(config_file))
        
        # Create scripts if requested
        if create_scripts:
            # Tokenization script
            tokenize_script = scripts_path / f"tokenize_{exp_info['config_name']}.sh"
            with open(tokenize_script, 'w') as f:
                f.write(create_tokenization_script(exp_info, str(base_path)))
            tokenize_script.chmod(0o755)
            print(f"  ✓ Created: {tokenize_script}")
            
            # Training script
            train_script = scripts_path / f"train_{exp_info['config_name']}.sh"
            with open(train_script, 'w') as f:
                f.write(create_training_script(exp_info, str(base_path)))
            train_script.chmod(0o755)
            print(f"  ✓ Created: {train_script}")
    
    print(f"\n✓ Generated {len(experiments)} experiment configurations!")
    print(f"Next steps:")
    print(f"  1. Review the generated configs in {output_path}")
    print(f"  2. Run tokenization: ./scripts/tokenize_<experiment>.sh")
    print(f"  3. Run training: ./scripts/train_<experiment>.sh")


if __name__ == "__main__":
    typer.run(main) 