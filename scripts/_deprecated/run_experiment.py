"""
Master Experiment Orchestration Script

This script runs the complete experimental pipeline for a given experiment number.
It handles data preprocessing, training, and evaluation in a coordinated manner.

Usage:
    python scripts/run_experiment.py <experiment_number> [options]
"""

import argparse
import sys
import yaml
from pathlib import Path
import subprocess
import time
from typing import Dict, List, Optional


class ExperimentOrchestrator:
    """Orchestrates the complete experimental pipeline."""
    
    def __init__(self, experiment_number: int, base_dir: Path, wandb_mode: str = "disabled"):
        """
        Initialize the experiment orchestrator.
        
        Args:
            experiment_number: The experiment number (0-7)
            base_dir: Project base directory
            wandb_mode: Weights & Biases mode (online, offline, or disabled)
        """
        self.experiment_number = experiment_number
        self.base_dir = base_dir
        self.wandb_mode = wandb_mode
        self.config_path = self._get_config_path()
        self.config = self._load_config()
        
    def _get_config_path(self) -> Path:
        """Get the configuration file path for the experiment."""
        if self.experiment_number == 0:
            return self.base_dir / "configs" / "experiment_0_baseline.yaml"
        else:
            return self.base_dir / "configs" / f"experiment_{self.experiment_number}.yaml"
    
    def _load_config(self) -> Dict:
        """Load the experiment configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def run_data_preprocessing(self) -> bool:
        """Run the data preprocessing pipeline."""
        print(f"\n=== Step 1: Data Preprocessing for Experiment {self.experiment_number} ===")
        
        # Check if we need to run ablations
        if 'dataset_manipulation' in self.config and self.config['dataset_manipulation']:
            print("Running corpus ablations...")
            for ablation in self.config['dataset_manipulation']:
                if not self._run_ablation(ablation):
                    print(f"❌ Ablation {ablation} failed")
                    return False
                print(f"✅ Ablation {ablation} completed")
        
        # Run data preprocessing
        print("Running data preprocessing...")
        result = subprocess.run([
            sys.executable, "-m", "model_foundry.cli", "preprocess-data", 
            str(self.config_path)
        ])
        
        if result.returncode != 0:
            print("❌ Data preprocessing failed")
            return False
        
        print("✅ Data preprocessing completed")
        return True
    
    def _run_ablation(self, ablation_name: str) -> bool:
        """Run a specific ablation script."""
        ablation_script = self.base_dir / "preprocessing" / f"{ablation_name}.py"
        
        if not ablation_script.exists():
            print(f"❌ Ablation script not found: {ablation_script}")
            return False
        
        # Determine input and output directories
        source_corpus = self.base_dir / self.config['data']['source_corpus']
        output_dir = self.base_dir / self.config['data']['training_corpus']
        
        # Run the ablation script
        result = subprocess.run([
            sys.executable, str(ablation_script),
            str(source_corpus), str(output_dir)
        ])
        
        return result.returncode == 0
    
    def run_tokenizer_training(self) -> bool:
        """Run tokenizer training."""
        print(f"\n=== Step 2: Tokenizer Training for Experiment {self.experiment_number} ===")
        
        result = subprocess.run([
            sys.executable, "-m", "model_foundry.cli", "train-tokenizer",
            str(self.config_path)
        ])
        
        if result.returncode != 0:
            print("❌ Tokenizer training failed")
            return False
        
        print("✅ Tokenizer training completed")
        return True
    
    def run_dataset_tokenization(self) -> bool:
        """Run dataset tokenization."""
        print(f"\n=== Step 3: Dataset Tokenization for Experiment {self.experiment_number} ===")
        
        result = subprocess.run([
            sys.executable, "-m", "model_foundry.cli", "tokenize-dataset",
            str(self.config_path)
        ])
        
        if result.returncode != 0:
            print("❌ Dataset tokenization failed")
            return False
        
        print("✅ Dataset tokenization completed")
        return True
    
    def run_data_chunking(self) -> bool:
        """Run data chunking."""
        print(f"\n=== Step 4: Data Chunking for Experiment {self.experiment_number} ===")
        
        result = subprocess.run([
            sys.executable, "-m", "model_foundry.cli", "preprocess-data",
            str(self.config_path)
        ])
        
        if result.returncode != 0:
            print("❌ Data chunking failed")
            return False
        
        print("✅ Data chunking completed")
        return True
    
    def generate_checkpoint_schedule(self) -> bool:
        """Generate checkpoint schedule."""
        print(f"\n=== Step 5: Checkpoint Schedule Generation for Experiment {self.experiment_number} ===")
        
        result = subprocess.run([
            sys.executable, "-m", "model_foundry.cli", "generate-checkpoints",
            str(self.config_path)
        ])
        
        if result.returncode != 0:
            print("❌ Checkpoint schedule generation failed")
            return False
        
        print("✅ Checkpoint schedule generation completed")
        return True
    
    def run_model_training(self) -> bool:
        """Run model training."""
        print(f"\n=== Step 6: Model Training for Experiment {self.experiment_number} ===")
        print(f"  - Weights & Biases mode: {self.wandb_mode}")
        
        # Set up environment variables for wandb
        import os
        env = os.environ.copy()
        
        if self.wandb_mode == "disabled":
            env["WANDB_MODE"] = "disabled"
            env["WANDB_SILENT"] = "true"
        elif self.wandb_mode == "offline":
            env["WANDB_MODE"] = "offline"
        else:  # online
            env["WANDB_MODE"] = "online"
        
        result = subprocess.run([
            sys.executable, "-m", "model_foundry.cli", "run",
            str(self.config_path)
        ], env=env)
        
        if result.returncode != 0:
            print("❌ Model training failed")
            return False
        
        print("✅ Model training completed")
        return True
    
    def run_evaluation(self, checkpoint_step: Optional[int] = None) -> bool:
        """Run model evaluation."""
        print(f"\n=== Step 7: Model Evaluation for Experiment {self.experiment_number} ===")
        
        # Get model and tokenizer paths
        model_dir = self.base_dir / self.config['training']['output_dir']
        tokenizer_dir = self.base_dir / self.config['tokenizer']['output_dir']
        
        if checkpoint_step is not None:
            model_path = model_dir / f"checkpoint-{checkpoint_step}"
        else:
            # Use the latest checkpoint
            model_path = model_dir
        
        # Run surprisal evaluation if stimuli exist
        stimuli_path = self.base_dir / "evaluation" / "stimuli" / "subject_drop_stimuli.json"
        if stimuli_path.exists():
            print("Running surprisal evaluation...")
            result = subprocess.run([
                sys.executable, "evaluation/surprisal.py",
                str(model_path), str(tokenizer_dir), str(stimuli_path),
                "--output", str(self.base_dir / "evaluation" / f"surprisal_exp{self.experiment_number}.json")
            ])
            
            if result.returncode != 0:
                print("❌ Surprisal evaluation failed")
                return False
            
            print("✅ Surprisal evaluation completed")
        
        # Run BLIMP evaluation
        print("Running BLIMP evaluation...")
        result = subprocess.run([
            sys.executable, "evaluation/run_blimp.py",
            str(model_path), str(tokenizer_dir),
            "--output", str(self.base_dir / "evaluation" / f"blimp_exp{self.experiment_number}.json")
        ])
        
        if result.returncode != 0:
            print("❌ BLIMP evaluation failed")
            return False
        
        print("✅ BLIMP evaluation completed")
        return True
    
    def run_full_pipeline(self, skip_steps: List[str] = None, 
                         checkpoint_step: Optional[int] = None) -> bool:
        """
        Run the complete experimental pipeline.
        
        Args:
            skip_steps: List of steps to skip
            checkpoint_step: Specific checkpoint to evaluate (if None, uses latest)
            
        Returns:
            True if all steps succeeded, False otherwise
        """
        if skip_steps is None:
            skip_steps = []
        
        steps = [
            ("data_preprocessing", self.run_data_preprocessing),
            ("tokenizer_training", self.run_tokenizer_training),
            ("dataset_tokenization", self.run_dataset_tokenization),
            ("data_chunking", self.run_data_chunking),
            ("checkpoint_schedule", self.generate_checkpoint_schedule),
            ("model_training", self.run_model_training),
            ("evaluation", lambda: self.run_evaluation(checkpoint_step))
        ]
        
        start_time = time.time()
        
        for step_name, step_func in steps:
            if step_name in skip_steps:
                print(f"\n⏭️  Skipping step: {step_name}")
                continue
            
            if not step_func():
                print(f"\n❌ Pipeline failed at step: {step_name}")
                return False
            
            print(f"✅ Step completed: {step_name}")
        
        total_time = time.time() - start_time
        print(f"\n🎉 Experiment {self.experiment_number} completed successfully!")
        print(f"Total time: {total_time/3600:.2f} hours")
        
        return True


def main():
    """Main function for experiment orchestration."""
    parser = argparse.ArgumentParser(description="Run complete experimental pipeline")
    parser.add_argument(
        "experiment_number",
        type=int,
        help="Experiment number (0-7)"
    )
    parser.add_argument(
        "--skip-steps",
        nargs="+",
        default=[],
        help="Steps to skip (data_preprocessing, tokenizer_training, etc.)"
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        help="Specific checkpoint step to evaluate (default: latest)"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Project base directory"
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="disabled",
        help="Weights & Biases mode (online, offline, or disabled)"
    )
    
    args = parser.parse_args()
    
    # Validate experiment number
    if args.experiment_number < 0 or args.experiment_number > 7:
        print("❌ Experiment number must be between 0 and 7")
        return 1
    
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"❌ Base directory does not exist: {base_dir}")
        return 1
    
    # Initialize orchestrator
    try:
        orchestrator = ExperimentOrchestrator(args.experiment_number, base_dir, args.wandb_mode)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    
    # Run the pipeline
    success = orchestrator.run_full_pipeline(
        skip_steps=args.skip_steps,
        checkpoint_step=args.checkpoint_step
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main()) 