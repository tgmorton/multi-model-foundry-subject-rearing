import argparse
import os
import yaml
import sentencepiece as spm
from datasets import load_dataset, disable_progress_bar
from pathlib import Path
import glob

# Correctly disable the progress bars from the datasets library
disable_progress_bar()


def find_project_root(start_path: str) -> str:
    """Finds the project root by searching upwards for a .git directory."""
    path = Path(start_path).resolve()
    while path.parent != path:
        if (path / '.git').is_dir():
            return str(path)
        path = path.parent
    print("Warning: .git directory not found. Falling back to current working directory as project root.")
    return os.getcwd()


def tokenize_dataset_from_config(config_path: str):
    """
    Loads training and test corpora, tokenizes them using the experiment-specific SentencePiece
    model, and saves the tokenized datasets to disk.
    """
    print(f"--- Tokenizing Dataset from Config: {config_path} ---")

    base_dir = find_project_root(__file__)
    print(f"  - Project Root (Base Directory): {base_dir}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    experiment_name = config.get('experiment_name', 'default_experiment')
    training_corpus_path_from_config = config['data']['training_corpus']
    test_corpus_path_from_config = config['data'].get('test_corpus')
    tokenizer_dir_from_config = config['tokenizer']['output_dir']

    training_corpus_path = training_corpus_path_from_config if os.path.isabs(training_corpus_path_from_config) else os.path.join(base_dir,
                                                                                                      training_corpus_path_from_config)
    tokenizer_dir = tokenizer_dir_from_config if os.path.isabs(tokenizer_dir_from_config) else os.path.join(base_dir,
                                                                                                            tokenizer_dir_from_config)
    tokenized_data_dir = os.path.join(base_dir, "data", "tokenized", experiment_name)
    tokenizer_model_path = os.path.join(tokenizer_dir, 'tokenizer.model')

    if not os.path.exists(training_corpus_path):
        print(f"FATAL ERROR: Training corpus path not found at '{training_corpus_path}'.")
        return
    if not os.path.exists(tokenizer_model_path):
        print(f"FATAL ERROR: Tokenizer model not found at '{tokenizer_model_path}'.")
        return

    print(f"  - Experiment:          {experiment_name}")
    print(f"  - Tokenizer Model:     {tokenizer_model_path}")
    print(f"  - Output Directory:    {tokenized_data_dir}")

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_model_path)
    print(f"  - Successfully loaded tokenizer with vocab size: {tokenizer.vocab_size()}")

    # Process training data
    print(f"\n  - Processing training data from '{training_corpus_path}'...")
    search_pattern = os.path.join(training_corpus_path, '**', '*.train')
    training_files = glob.glob(search_pattern, recursive=True)

    if not training_files:
        print(f"FATAL ERROR: No .train files found in '{training_corpus_path}'.")
        return

    print(f"  - Loading {len(training_files)} training file(s)...")
    # Use custom cache directory to avoid filling up home directory
    # Can be overridden with HF_DATASETS_CACHE environment variable
    cache_dir = os.environ.get('HF_DATASETS_CACHE', os.path.join(tokenized_data_dir, '.cache'))
    os.makedirs(cache_dir, exist_ok=True)
    print(f"  - Using cache directory: {cache_dir}")
    raw_training_dataset = load_dataset('text', data_files={'train': training_files}, split='train', cache_dir=cache_dir)
    print(f"  - Found {len(raw_training_dataset):,} total lines in the training corpus.")

    def tokenize_function(examples):
        return {'input_ids': tokenizer.encode(examples['text'], out_type=int)}

    print("  - Tokenizing training dataset (this may take a while)...")
    tokenized_training_dataset = raw_training_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=['text']
    )
    print("  - Training tokenization complete.")

    # Process test data if available
    tokenized_test_dataset = None
    if test_corpus_path_from_config:
        test_corpus_path = test_corpus_path_from_config if os.path.isabs(test_corpus_path_from_config) else os.path.join(base_dir, test_corpus_path_from_config)
        
        if os.path.exists(test_corpus_path):
            print(f"\n  - Processing test data from '{test_corpus_path}'...")
            search_pattern = os.path.join(test_corpus_path, '**', '*.test')
            test_files = glob.glob(search_pattern, recursive=True)

            if test_files:
                print(f"  - Loading {len(test_files)} test file(s)...")
                raw_test_dataset = load_dataset('text', data_files={'test': test_files}, split='test', cache_dir=cache_dir)
                print(f"  - Found {len(raw_test_dataset):,} total lines in the test corpus.")

                print("  - Tokenizing test dataset...")
                tokenized_test_dataset = raw_test_dataset.map(
                    tokenize_function,
                    batched=True,
                    num_proc=os.cpu_count(),
                    remove_columns=['text']
                )
                print("  - Test tokenization complete.")
            else:
                print(f"  - Warning: No .test files found in '{test_corpus_path}'.")
        else:
            print(f"  - Warning: Test corpus path not found at '{test_corpus_path}'.")

    # Save datasets
    print(f"\n  - Saving tokenized datasets to '{tokenized_data_dir}'...")
    os.makedirs(tokenized_data_dir, exist_ok=True)
    
    # Save training dataset
    tokenized_training_dataset.save_to_disk(os.path.join(tokenized_data_dir, 'train'))
    
    # Save test dataset if available
    if tokenized_test_dataset:
        tokenized_test_dataset.save_to_disk(os.path.join(tokenized_data_dir, 'test'))
        print("  - Saved both training and test datasets.")
    else:
        print("  - Saved training dataset only (no test data available).")
    
    print("\n----- Dataset Tokenization Complete -----")


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize a dataset for a specific experiment using its trained tokenizer."
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the experiment's .yaml configuration file."
    )
    args = parser.parse_args()

    project_root = find_project_root(__file__)
    absolute_config_path = args.config_path if os.path.isabs(args.config_path) else os.path.join(project_root,
                                                                                                 args.config_path)

    if not os.path.exists(absolute_config_path):
        print(f"FATAL ERROR: Configuration file not found at '{absolute_config_path}'")
        return

    tokenize_dataset_from_config(absolute_config_path)


if __name__ == '__main__':
    main()