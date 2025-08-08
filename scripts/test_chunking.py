#!/usr/bin/env python3
"""
Test script to analyze chunking behavior for both exp0 and exp1 datasets.
This script provides verbose debugging information to understand why the chunk counts differ.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
import sentencepiece as spm
import argparse
from typing import Dict, List, Tuple
import gc
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_foundry.config import Config
from model_foundry.data import DataProcessor


def analyze_tokenized_dataset(dataset: Dataset, name: str) -> Dict:
    """Analyze a tokenized dataset and return statistics."""
    print(f"\n{'='*60}")
    print(f"Analyzing tokenized dataset: {name}")
    print(f"{'='*60}")
    
    input_ids = dataset['input_ids']
    
    # Calculate sequence length statistics
    lengths = [len(seq) for seq in input_ids]
    
    stats = {
        'name': name,
        'num_sequences': len(input_ids),
        'total_tokens': sum(lengths),
        'avg_length': np.mean(lengths),
        'median_length': np.median(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'std_length': np.std(lengths),
    }
    
    # Calculate length distribution
    length_bins = [0, 100, 250, 500, 750, 1000, 1500, 2000, 5000, 10000, float('inf')]
    length_dist = {}
    
    for i in range(len(length_bins) - 1):
        lower, upper = length_bins[i], length_bins[i+1]
        count = sum(1 for l in lengths if lower <= l < upper)
        bin_label = f"{lower}-{upper}" if upper != float('inf') else f"{lower}+"
        length_dist[bin_label] = count
    
    stats['length_distribution'] = length_dist
    
    # Print statistics
    print(f"Total sequences: {stats['num_sequences']:,}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Average length: {stats['avg_length']:.1f}")
    print(f"Median length: {stats['median_length']:.1f}")
    print(f"Min/Max length: {stats['min_length']} / {stats['max_length']}")
    print(f"Std deviation: {stats['std_length']:.1f}")
    
    print("\nLength distribution:")
    for bin_label, count in length_dist.items():
        if count > 0:
            pct = (count / len(lengths)) * 100
            print(f"  {bin_label:10s}: {count:7,} ({pct:5.1f}%)")
    
    return stats


def simulate_chunking(dataset: Dataset, chunk_size: int, method: str = "current") -> Tuple[int, Dict]:
    """
    Simulate the chunking process and return the number of chunks.
    
    Args:
        dataset: Tokenized dataset
        chunk_size: Target chunk size
        method: Either 'current' (old method) or 'concatenate' (new method)
    """
    print(f"\n{'-'*50}")
    print(f"Simulating {method} chunking method (chunk_size={chunk_size})")
    print(f"{'-'*50}")
    
    input_ids = dataset['input_ids']
    total_sequences = len(input_ids)
    total_tokens = sum(len(seq) for seq in input_ids)
    
    if method == "current":
        # Old method: chunk each sequence independently
        chunks_created = 0
        tokens_used = 0
        tokens_wasted = 0
        sequences_skipped = 0
        
        for sequence in input_ids:
            seq_len = len(sequence)
            if seq_len < chunk_size:
                sequences_skipped += 1
                tokens_wasted += seq_len
            else:
                num_chunks = seq_len // chunk_size
                chunks_created += num_chunks
                tokens_used += num_chunks * chunk_size
                tokens_wasted += seq_len % chunk_size
        
        stats = {
            'chunks_created': chunks_created,
            'tokens_used': tokens_used,
            'tokens_wasted': tokens_wasted,
            'sequences_skipped': sequences_skipped,
            'sequences_processed': total_sequences - sequences_skipped,
            'efficiency': (tokens_used / total_tokens) * 100 if total_tokens > 0 else 0
        }
        
    else:  # concatenate method
        # New method: concatenate all sequences then chunk
        chunks_created = 0
        concatenated_length = total_tokens
        
        chunks_created = concatenated_length // chunk_size
        tokens_used = chunks_created * chunk_size
        tokens_wasted = concatenated_length % chunk_size
        
        stats = {
            'chunks_created': chunks_created,
            'tokens_used': tokens_used,
            'tokens_wasted': tokens_wasted,
            'sequences_skipped': 0,
            'sequences_processed': total_sequences,
            'efficiency': (tokens_used / total_tokens) * 100 if total_tokens > 0 else 0
        }
    
    # Print results
    print(f"Total sequences: {total_sequences:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Chunks created: {stats['chunks_created']:,}")
    print(f"Tokens used: {stats['tokens_used']:,}")
    print(f"Tokens wasted: {stats['tokens_wasted']:,}")
    if method == "current":
        print(f"Sequences skipped: {stats['sequences_skipped']:,} ({(stats['sequences_skipped']/total_sequences)*100:.1f}%)")
    print(f"Token efficiency: {stats['efficiency']:.1f}%")
    
    return chunks_created, stats


def test_actual_chunking(config: Config) -> Dict:
    """Test the actual chunking implementation from DataProcessor."""
    print(f"\n{'='*60}")
    print(f"Testing actual DataProcessor chunking for: {config.experiment_name}")
    print(f"{'='*60}")
    
    processor = DataProcessor(config)
    
    # Check if tokenized dataset exists
    if not os.path.exists(processor.tokenized_data_dir):
        print(f"❌ Tokenized dataset not found at: {processor.tokenized_data_dir}")
        return None
    
    # Load tokenized dataset
    try:
        tokenized_dataset = load_from_disk(processor.tokenized_data_dir)
        print(f"✓ Loaded tokenized dataset from: {processor.tokenized_data_dir}")
        print(f"  - Size: {len(tokenized_dataset):,} sequences")
    except Exception as e:
        print(f"❌ Error loading tokenized dataset: {e}")
        return None
    
    # Remove old chunked data if it exists
    if os.path.exists(processor.chunked_data_dir):
        print(f"  - Removing existing chunked data at: {processor.chunked_data_dir}")
        import shutil
        shutil.rmtree(processor.chunked_data_dir)
    
    # Run the actual chunking
    print(f"\n  - Running actual chunking process...")
    success = processor.preprocess_data(force_reprocess=True)
    
    if not success:
        print(f"❌ Chunking failed")
        return None
    
    # Load and analyze the chunked dataset
    try:
        chunked_dataset = load_from_disk(processor.chunked_data_dir)
        print(f"\n✓ Chunked dataset created:")
        print(f"  - Chunks: {len(chunked_dataset):,}")
        print(f"  - Chunk size: {config.data.max_sequence_length}")
        
        # Calculate steps per epoch
        effective_batch_size = config.data.batch_size * config.training.gradient_accumulation_steps
        steps_per_epoch = len(chunked_dataset) // effective_batch_size
        
        print(f"\n  Training implications:")
        print(f"  - Batch size: {config.data.batch_size}")
        print(f"  - Gradient accumulation: {config.training.gradient_accumulation_steps}")
        print(f"  - Effective batch size: {effective_batch_size}")
        print(f"  - Steps per epoch: {steps_per_epoch}")
        print(f"  - Total training steps (20 epochs): {steps_per_epoch * 20}")
        
        return {
            'num_chunks': len(chunked_dataset),
            'steps_per_epoch': steps_per_epoch,
            'total_steps': steps_per_epoch * 20
        }
        
    except Exception as e:
        print(f"❌ Error loading chunked dataset: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Test chunking behavior for experiments')
    parser.add_argument('--exp0-config', type=str, default='configs/experiment_0_baseline.yaml',
                       help='Path to exp0 config file')
    parser.add_argument('--exp1-config', type=str, default='configs/experiment_1_remove_expletives.yaml',
                       help='Path to exp1 config file')
    parser.add_argument('--chunk-size', type=int, default=1000,
                       help='Chunk size for testing (default: 1000)')
    parser.add_argument('--test-actual', action='store_true',
                       help='Test actual DataProcessor chunking (will recreate chunks)')
    args = parser.parse_args()
    
    # Load configurations
    exp0_config = Config.from_yaml(args.exp0_config)
    exp1_config = Config.from_yaml(args.exp1_config)
    
    print("="*80)
    print("CHUNKING ANALYSIS AND DEBUGGING")
    print("="*80)
    
    results = {}
    
    # Analyze both experiments
    for config, exp_name in [(exp0_config, 'exp0_baseline'), (exp1_config, 'exp1_remove_expletives')]:
        print(f"\n{'#'*80}")
        print(f"# EXPERIMENT: {exp_name}")
        print(f"{'#'*80}")
        
        # Try to load tokenized dataset
        tokenized_dir = os.path.join(config.base_dir, "data", "tokenized", config.experiment_name)
        
        if not os.path.exists(tokenized_dir):
            print(f"❌ Tokenized dataset not found at: {tokenized_dir}")
            print(f"   Please run tokenization first.")
            continue
        
        try:
            tokenized_dataset = load_from_disk(tokenized_dir)
            print(f"✓ Loaded tokenized dataset from: {tokenized_dir}")
            
            # Analyze the tokenized dataset
            stats = analyze_tokenized_dataset(tokenized_dataset, exp_name)
            
            # Simulate both chunking methods
            print(f"\n{'='*60}")
            print(f"CHUNKING SIMULATION")
            print(f"{'='*60}")
            
            current_chunks, current_stats = simulate_chunking(
                tokenized_dataset, args.chunk_size, method="current"
            )
            
            concat_chunks, concat_stats = simulate_chunking(
                tokenized_dataset, args.chunk_size, method="concatenate"
            )
            
            # Compare the two methods
            print(f"\n{'='*60}")
            print(f"CHUNKING METHOD COMPARISON")
            print(f"{'='*60}")
            print(f"Current method: {current_chunks:,} chunks")
            print(f"Concatenate method: {concat_chunks:,} chunks")
            print(f"Difference: {concat_chunks - current_chunks:,} chunks ({((concat_chunks/current_chunks - 1) * 100):.1f}% more)")
            
            results[exp_name] = {
                'tokenized_stats': stats,
                'current_method': current_stats,
                'concat_method': concat_stats,
                'current_chunks': current_chunks,
                'concat_chunks': concat_chunks
            }
            
            # Test actual chunking if requested
            if args.test_actual:
                actual_results = test_actual_chunking(config)
                if actual_results:
                    results[exp_name]['actual'] = actual_results
            
        except Exception as e:
            print(f"❌ Error processing {exp_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final comparison
    if len(results) == 2:
        print(f"\n{'#'*80}")
        print(f"# FINAL COMPARISON")
        print(f"{'#'*80}")
        
        exp0_data = results.get('exp0_baseline', {})
        exp1_data = results.get('exp1_remove_expletives', {})
        
        if exp0_data and exp1_data:
            print("\nTokenized Dataset Comparison:")
            print(f"  exp0 total tokens: {exp0_data['tokenized_stats']['total_tokens']:,}")
            print(f"  exp1 total tokens: {exp1_data['tokenized_stats']['total_tokens']:,}")
            print(f"  Difference: {abs(exp0_data['tokenized_stats']['total_tokens'] - exp1_data['tokenized_stats']['total_tokens']):,}")
            
            print("\nCurrent Chunking Method:")
            print(f"  exp0: {exp0_data['current_chunks']:,} chunks")
            print(f"  exp1: {exp1_data['current_chunks']:,} chunks")
            print(f"  Ratio: {exp0_data['current_chunks'] / exp1_data['current_chunks']:.2f}x")
            
            print("\nConcatenate Chunking Method:")
            print(f"  exp0: {exp0_data['concat_chunks']:,} chunks")
            print(f"  exp1: {exp1_data['concat_chunks']:,} chunks")
            print(f"  Ratio: {exp0_data['concat_chunks'] / exp1_data['concat_chunks']:.2f}x")
            
            if 'actual' in exp0_data and 'actual' in exp1_data:
                print("\nActual Implementation Results:")
                print(f"  exp0: {exp0_data['actual']['num_chunks']:,} chunks, {exp0_data['actual']['steps_per_epoch']} steps/epoch")
                print(f"  exp1: {exp1_data['actual']['num_chunks']:,} chunks, {exp1_data['actual']['steps_per_epoch']} steps/epoch")
    
    # Save results to file
    output_file = "chunking_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()