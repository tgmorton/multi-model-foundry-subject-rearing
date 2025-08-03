"""
Lemmatize Verbs Ablation Script

This script lemmatizes all verbs to their infinitive form in the corpus.
This ablation tests how the model learns when verb morphology is impoverished.

Procedure: LemmatizeVerbs(text)
1. Load spaCy NLP model with POS tagger and lemmatizer
2. Initialize modified_parts an empty list
3. doc = process(text, NLP model)
4. for each token in doc do
5.   if token.pos_ == 'VERB':
6.     append token.lemma_ + token.whitespace_ to modified_parts
7.   else:
8.     append token.text_with_ws to modified_parts
9.   end if
10. end for
11. result = join(modified_parts)
12. return result
"""

import argparse
import spacy
import random
import os
import glob
from tqdm import tqdm
import math
import logging
from datetime import datetime
import sys
from pathlib import Path

# Add model_foundry to path for logging_utils import
sys.path.insert(0, str(Path(__file__).parent.parent / "model_foundry"))
from logging_utils import setup_logging


def get_spacy_device(verbose=False):
    """
    Detects and returns the best available spaCy device.
    Checks for Apple Silicon (MPS), then CUDA, and defaults to CPU.
    """
    # Debug: Check environment variables
    import os
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    if verbose:
        print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    
    try:
        import torch
        if verbose:
            print(f"PyTorch version: {torch.__version__}")
        
        # Check for Apple Silicon (MPS)
        if torch.backends.mps.is_available():
            if verbose:
                print("Apple Silicon (MPS) device detected. Using GPU.")
            return "mps"
        
        # Check for CUDA
        if torch.cuda.is_available():
            if verbose:
                print(f"NVIDIA CUDA device detected. Using GPU.")
                print(f"CUDA device count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            return "cuda"
        else:
            if verbose:
                print("PyTorch CUDA not available")
            
    except ImportError as e:
        if verbose:
            print(f"PyTorch import error: {e}")
    except Exception as e:
        if verbose:
            print(f"GPU detection error: {e}")

    if verbose:
        print("Warning: No compatible GPU detected. spaCy will run on CPU, which may be slow.")
    return "cpu"


def count_tokens(text):
    """Count tokens using a more sophisticated approach that better matches training tokenization."""
    import re
    # Use regex to split on whitespace, handling punctuation better
    tokens = re.findall(r'\S+', text)
    return len(tokens)


def lemmatize_verbs_doc(doc, logger=None):
    """
    Lemmatize all verbs to their infinitive form in a spaCy Doc object.
    
    Args:
        doc: spaCy Doc object to process
        logger: Optional logger for debugging
        
    Returns:
        Tuple of (ablated_text, num_lemmatized)
    """
    modified_parts = []
    num_lemmatized = 0
    
    for token in doc:
        if token.pos_ == "VERB":
            modified_parts.append(token.lemma_ + token.whitespace_)
            num_lemmatized += 1
            if logger:
                logger.debug(f"Lemmatized verb: '{token.text}' -> '{token.lemma_}'")
        else:
            modified_parts.append(token.text_with_ws)
    
    result = ''.join(modified_parts)
    return result, num_lemmatized


def ablate_doc(doc, logger=None):
    """
    Performs the ablation on a single spaCy Doc object.
    Returns the ablated text for that doc.
    """
    return lemmatize_verbs_doc(doc, logger)


# Removed local setup_logging function - now using unified logging from model_foundry.logging_utils


def validate_verb_lemmatization(original_text, ablated_text, nlp):
    """
    Validate that verbs were actually lemmatized.
    
    Args:
        original_text: Original text before ablation
        ablated_text: Text after ablation
        nlp: spaCy NLP model
        
    Returns:
        bool: True if verbs were found and lemmatized
    """
    original_doc = nlp(original_text)
    ablated_doc = nlp(ablated_text)
    
    original_verbs = [token.text for token in original_doc if token.pos_ == 'VERB']
    ablated_verbs = [token.text for token in ablated_doc if token.pos_ == 'VERB']
    
    if original_verbs:
        print(f"  Found {len(original_verbs)} verbs in original text")
        print(f"  Found {len(ablated_verbs)} verbs in ablated text")
        # Check if any verbs were lemmatized (different forms)
        original_verb_forms = set(original_verbs)
        ablated_verb_forms = set(ablated_verbs)
        lemmatized_count = len(original_verb_forms - ablated_verb_forms)
        print(f"  {lemmatized_count} unique verb forms were lemmatized")
        return lemmatized_count > 0
    else:
        print(f"  No verbs found in original text")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Lemmatize verbs in a directory of .train files while maintaining dataset size."
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing source .train files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the ablated .train files.")
    parser.add_argument("--replacement_pool_dir", required=True,
                        help="Directory containing corresponding replacement pool files.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Number of lines to process in each chunk.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging to logs directory.")
    parser.add_argument("--skip_validation", action="store_true", help="Skip verb lemmatization validation to speed up processing.")
    args = parser.parse_args()

    # --- Device and Model Loading ---
    spacy_device = get_spacy_device(verbose=args.verbose)
    spacy.prefer_gpu(spacy_device)

    if args.verbose:
        print("Loading spaCy models...")
    # Use the smaller model for faster processing on Windows
    try:
        nlp = spacy.load("en_core_web_sm")
        if args.verbose:
            print("Using en_core_web_sm for faster processing")
    except OSError:
        if args.verbose:
            print("en_core_web_sm not found, trying en_core_web_trf...")
        nlp = spacy.load("en_core_web_trf")
        if args.verbose:
            print("Using en_core_web_trf (slower but more accurate)")
    
    # Increase max_length to handle larger texts
    nlp.max_length = 2000000  # 2M characters instead of default 1M

    if args.verbose:
        print("Models loaded successfully.")

    # --- Set up logging if verbose mode is enabled ---
    logger = None
    if args.verbose:
        experiment_name = os.path.basename(args.output_dir)
        logger = setup_logging(__name__, experiment=experiment_name, log_dir="logs")
        logger.info("=== Starting verb lemmatization with verbose logging ===")
        logger.info(f"Input directory: {args.input_dir}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Replacement pool directory: {args.replacement_pool_dir}")
        logger.info(f"Chunk size: {args.chunk_size}")

    # --- Find all source files ---
    search_pattern = os.path.join(args.input_dir, '**', '*.train')
    source_files = glob.glob(search_pattern, recursive=True)
    if not source_files:
        raise FileNotFoundError(f"No '.train' files found in {args.input_dir}")

    if logger:
        logger.info(f"Found {len(source_files)} source files to process")

    # --- Process each file individually ---
    for source_path in tqdm(source_files, desc="Processing Corpus Files"):
        
        # --- File-specific stats ---
        stats = {
            "file_name": os.path.basename(source_path),
            "original_tokens": 0,
            "tokens_removed": 0,
            "verbs_lemmatized": 0,
        }

        relative_path = os.path.relpath(source_path, args.input_dir)
        pool_path = os.path.join(args.replacement_pool_dir, relative_path)
        output_path = os.path.join(args.output_dir, relative_path)

        if not os.path.exists(pool_path):
            print(f"\nWarning: No matching pool file found for {source_path}. Skipping.")
            continue

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(source_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Calculate target token count from the full original text
            original_text = "".join(lines)
            stats["original_tokens"] = count_tokens(original_text)
            target_token_count = stats["original_tokens"]

        if target_token_count == 0:
            open(output_path, 'w').close()
            continue

        with open(pool_path, 'r', encoding='utf-8') as f:
            replacement_pool_sentences = f.readlines()

        if logger:
            logger.info(f"\n=== Processing file: {os.path.basename(source_path)} ===")
            logger.info(f"File path: {source_path}")
            logger.info(f"Number of lines: {len(lines)}")
            logger.info(f"Target token count: {target_token_count}")

        # --- Ablate the main file in chunks with a correct progress bar ---
        ablated_text = ""
        original_text = "".join(lines)
        
        with tqdm(
                total=len(lines), desc=f"  Ablating {os.path.basename(source_path)}", leave=False
        ) as pbar:
            # Manually create batches to have precise control over the progress update
            for i in range(0, len(lines), args.chunk_size):
                chunk = lines[i:i + args.chunk_size]
                docs = nlp.pipe(chunk)
                for doc in docs:
                    ablated_doc_text, num_lemmatized = ablate_doc(doc, logger)
                    ablated_text += ablated_doc_text
                    stats["verbs_lemmatized"] += num_lemmatized
                pbar.update(len(chunk))  # Update by the actual number of lines in the chunk
                pbar.set_postfix(lemmatized=f'{stats["verbs_lemmatized"]:,}')
        
        # Validate that verbs were actually lemmatized (unless skipped)
        if not args.skip_validation:
            print(f"\n  Validating verb lemmatization for {os.path.basename(source_path)}:")
            validation_success = validate_verb_lemmatization(original_text, ablated_text, nlp)
            if not validation_success:
                print(f"  ⚠️  Warning: No verbs were lemmatized from {os.path.basename(source_path)}")
        else:
            print(f"\n  Skipping validation for {os.path.basename(source_path)} (--skip_validation flag used)")

        # --- Iterative Replacement Loop ---
        current_token_count = count_tokens(ablated_text)
        
        # Add a progress bar for the replacement loop
        with tqdm(total=target_token_count, initial=current_token_count, desc="  Rebuilding to size", leave=False) as pbar_rebuild:
            while current_token_count < target_token_count:
                tokens_needed = target_token_count - current_token_count
                
                # Estimate how many sentences to grab from the pool
                # Use a small sample to estimate average sentence length
                sample_sentences = random.sample(replacement_pool_sentences, k=min(10, len(replacement_pool_sentences)))
                avg_tokens_per_sentence = count_tokens(" ".join(sample_sentences)) / len(sample_sentences) if sample_sentences else 20
                
                # Grab a chunk of sentences to process
                num_sentences_to_grab = math.ceil(tokens_needed / avg_tokens_per_sentence) if avg_tokens_per_sentence > 0 else 100
                
                text_to_add_chunk_list = []
                for _ in range(num_sentences_to_grab):
                    if not replacement_pool_sentences:
                        break
                    text_to_add_chunk_list.append(replacement_pool_sentences.pop(random.randint(0, len(replacement_pool_sentences) - 1)))
                
                text_to_add_chunk = "".join(text_to_add_chunk_list)

                if not text_to_add_chunk:
                    print(f"\nWarning: Replacement pool for {source_path} exhausted.")
                    break

                ablated_chunk_to_add = ""
                chunk_tokens = 0
                for doc in nlp.pipe(text_to_add_chunk.splitlines()):
                    ablated_line, num_lemmatized = ablate_doc(doc, logger if args.verbose else None)
                    ablated_chunk_to_add += ablated_line
                    chunk_tokens += count_tokens(ablated_line)
                    stats["verbs_lemmatized"] += num_lemmatized

                ablated_text += ablated_chunk_to_add
                
                # Update the progress bar and token count
                pbar_rebuild.update(chunk_tokens)
                current_token_count += chunk_tokens
                pbar_rebuild.set_postfix(lemmatized=f'{stats["verbs_lemmatized"]:,}')

        # Write the ablated text to the output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(ablated_text.splitlines(keepends=True))
            
        # --- Save remainder of replacement pool ---
        remainder_dir = os.path.join(args.output_dir, "replacement_pool_remainder")
        os.makedirs(remainder_dir, exist_ok=True)
        # Remove .train extension before adding .txt
        base_name = os.path.basename(source_path)
        if base_name.endswith('.train'):
            base_name = base_name[:-6]  # Remove '.train'
        remainder_path = os.path.join(remainder_dir, f"{base_name}.txt")
        
        with open(remainder_path, 'w', encoding='utf-8') as f:
            f.writelines(replacement_pool_sentences)
        
        print(f"  ✓ Saved replacement pool remainder to {remainder_path}")
            
        # --- Save stats to JSON file ---
        stats["tokens_removed"] = stats["original_tokens"] - count_tokens(ablated_text)
        if stats["original_tokens"] > 0:
            stats["proportion_removed"] = stats["tokens_removed"] / stats["original_tokens"]
        else:
            stats["proportion_removed"] = 0
            
        stats_dir = os.path.join(args.output_dir, "statistics")
        os.makedirs(stats_dir, exist_ok=True)
        stats_filename = os.path.join(stats_dir, f"{os.path.basename(source_path)}.json")
        
        with open(stats_filename, 'w', encoding='utf-8') as f:
            import json
            json.dump(stats, f, indent=4)
        
        print(f"  ✓ Saved statistics to {stats_filename}")

    print("\nAblation complete for all files.")


if __name__ == "__main__":
    main()
