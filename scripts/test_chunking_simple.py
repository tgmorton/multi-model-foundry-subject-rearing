#!/usr/bin/env python3
"""
Simple script to test chunking for a single experiment config.
Chunks the tokenized data to a temporary folder and reports statistics.
"""

import os
import sys
import argparse
import tempfile
import shutil
from pathlib import Path
from datasets import load_from_disk
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_foundry.config import ExperimentConfig
from model_foundry.data import DataProcessor


def analyze_chunking(config_path: str):
    """
    Load a config, chunk its tokenized data to a temp folder, and report stats.
    """
    print("="*80)
    print(f"CHUNKING TEST FOR: {config_path}")
    print("="*80)
    
    # Load configuration
    config = ExperimentConfig.from_yaml(config_path)
    print(f"\nExperiment: {config.experiment_name}")
    print(f"Base dir: {config.base_dir}")
    
    # Create a temporary directory for chunked data
    with tempfile.TemporaryDirectory(prefix=f"chunk_test_{config.experiment_name}_") as temp_dir:
        print(f"\nUsing temporary directory: {temp_dir}")
        
        # Create a modified DataProcessor that uses the temp directory
        processor = DataProcessor(config)
        
        # Check if tokenized dataset exists
        if not os.path.exists(processor.tokenized_data_dir):
            print(f"\n❌ ERROR: Tokenized dataset not found at: {processor.tokenized_data_dir}")
            print(f"   Please run tokenization first.")
            return
        
        # Load tokenized dataset
        try:
            tokenized_dataset = load_from_disk(processor.tokenized_data_dir)
            print(f"\n✓ Loaded tokenized dataset from: {processor.tokenized_data_dir}")
            print(f"  - Number of sequences: {len(tokenized_dataset):,}")
            
            # Analyze tokenized dataset
            sequences = tokenized_dataset['input_ids']
            lengths = [len(seq) for seq in sequences]
            total_tokens = sum(lengths)
            
            print(f"\nTokenized Dataset Statistics:")
            print(f"  - Total sequences: {len(sequences):,}")
            print(f"  - Total tokens: {total_tokens:,}")
            print(f"  - Average sequence length: {np.mean(lengths):.1f}")
            print(f"  - Median sequence length: {np.median(lengths):.1f}")
            print(f"  - Min/Max length: {min(lengths)} / {max(lengths)}")
            print(f"  - Std deviation: {np.std(lengths):.1f}")
            
            # Length distribution
            print(f"\n  Length distribution:")
            bins = [0, 100, 250, 500, 750, 1000, 1500, 2000, 5000, 10000, float('inf')]
            for i in range(len(bins) - 1):
                lower, upper = bins[i], bins[i+1]
                count = sum(1 for l in lengths if lower <= l < upper)
                if count > 0:
                    pct = (count / len(lengths)) * 100
                    bin_label = f"{lower}-{upper}" if upper != float('inf') else f"{lower}+"
                    print(f"    {bin_label:10s}: {count:7,} sequences ({pct:5.1f}%)")
            
        except Exception as e:
            print(f"\n❌ ERROR loading tokenized dataset: {e}")
            return
        
        # Override the chunked data directory to use temp folder
        processor.chunked_data_dir = os.path.join(temp_dir, "chunked")
        
        print(f"\n" + "="*60)
        print("CHUNKING PROCESS")
        print("="*60)
        print(f"Chunk size: {config.data.max_sequence_length}")
        print(f"Output directory: {processor.chunked_data_dir}")
        
        # Run the chunking process
        print(f"\nStarting chunking process...")
        success = processor.preprocess_data(force_reprocess=True)
        
        if not success:
            print(f"\n❌ ERROR: Chunking failed")
            return
        
        # Load and analyze the chunked dataset
        try:
            chunked_dataset = load_from_disk(processor.chunked_data_dir)
            num_chunks = len(chunked_dataset)
            
            print(f"\n✓ Chunking completed successfully!")
            print(f"\n" + "="*60)
            print("CHUNKING RESULTS")
            print("="*60)
            print(f"Number of chunks created: {num_chunks:,}")
            
            # Calculate training implications
            batch_size = config.data.batch_size
            grad_accum = config.training.gradient_accumulation_steps
            effective_batch_size = batch_size * grad_accum
            steps_per_epoch = num_chunks // effective_batch_size
            remainder = num_chunks % effective_batch_size
            
            print(f"\n" + "="*60)
            print("TRAINING IMPLICATIONS")
            print("="*60)
            print(f"Batch size: {batch_size}")
            print(f"Gradient accumulation steps: {grad_accum}")
            print(f"Effective batch size: {effective_batch_size}")
            print(f"Steps per epoch: {steps_per_epoch}")
            print(f"Remainder chunks (unused per epoch): {remainder}")
            print(f"Training epochs configured: {config.training.epochs}")
            print(f"Total training steps: {steps_per_epoch * config.training.epochs}")
            
            # Memory estimation
            print(f"\n" + "="*60)
            print("MEMORY USAGE ESTIMATION")
            print("="*60)
            data_per_step = (num_chunks / steps_per_epoch) if steps_per_epoch > 0 else num_chunks
            data_percentage = (data_per_step / num_chunks) * 100
            print(f"Chunks processed per step: {data_per_step:.1f}")
            print(f"Percentage of dataset per step: {data_percentage:.1f}%")
            
            if data_percentage > 10:
                print(f"\n⚠️  WARNING: Processing {data_percentage:.1f}% of dataset per step!")
                print(f"   This may cause OOM errors. Consider:")
                print(f"   - Reducing batch_size (currently {batch_size})")
                print(f"   - Reducing gradient_accumulation_steps (currently {grad_accum})")
            
            # Efficiency analysis
            print(f"\n" + "="*60)
            print("CHUNKING EFFICIENCY")
            print("="*60)
            tokens_in_chunks = num_chunks * config.data.max_sequence_length
            efficiency = (tokens_in_chunks / total_tokens) * 100 if total_tokens > 0 else 0
            wasted_tokens = total_tokens - tokens_in_chunks
            
            print(f"Total tokens in original data: {total_tokens:,}")
            print(f"Total tokens in chunks: {tokens_in_chunks:,}")
            print(f"Tokens wasted: {abs(wasted_tokens):,}")
            print(f"Chunking efficiency: {min(100, efficiency):.1f}%")
            
            if efficiency > 100:
                print(f"\nNote: Efficiency >100% indicates padding was added to reach chunk size.")
            
        except Exception as e:
            print(f"\n❌ ERROR loading chunked dataset: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print(f"\n" + "="*80)
        print("TEST COMPLETED")
        print("="*80)
        print(f"Temporary chunked data was created in: {temp_dir}")
        print(f"This directory will be automatically cleaned up.")
        print(f"\nTo apply this chunking permanently, run the preprocessing")
        print(f"pipeline for experiment: {config.experiment_name}")


def main():
    parser = argparse.ArgumentParser(description='Test chunking for a single experiment')
    parser.add_argument('config', type=str, help='Path to the experiment config file')
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"❌ ERROR: Config file not found: {args.config}")
        sys.exit(1)
    
    analyze_chunking(args.config)


if __name__ == "__main__":
    main()