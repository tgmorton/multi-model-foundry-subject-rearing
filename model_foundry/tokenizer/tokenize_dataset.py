import argparse
import hashlib
import json
import os
import yaml
import sentencepiece as spm
from datasets import load_dataset, disable_progress_bar
from pathlib import Path
import glob

from model_foundry.cache_keys import (
    compute_cache_key,
    cache_meta,
    resolve_cached_path,
)

# Correctly disable the progress bars from the datasets library
disable_progress_bar()

# Name of the compose manifest emitted by the ablation/compose pipeline
# alongside the real top-level *.train files in a manipulation corpus dir.
COMPOSE_MANIFEST_NAME = "COMPOSE_MANIFEST.json"


def find_project_root(start_path: str) -> str:
    """Finds the project root by searching upwards for a .git directory."""
    path = Path(start_path).resolve()
    while path.parent != path:
        if (path / '.git').is_dir():
            return str(path)
        path = path.parent
    print("Warning: .git directory not found. Falling back to current working directory as project root.")
    return os.getcwd()


def _stream_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    """Stream SHA-256 of a file; safe for arbitrarily large corpus files."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _resolve_training_files(training_corpus_path: str) -> list:
    """Resolve the exact list of *.train files to ingest for a corpus dir.

    This is the integrity gate that replaced a recursive ``**/*.train``
    glob. On 2026-06-04 that recursive glob silently ingested build
    INTERMEDIATES (``_train/``, ``_pool/``, ``pool_remainder/`` subdirs)
    nested inside every manipulation corpus directory, producing ~2.2x
    duplicated training data — a study-breaking contamination incident.
    Ingestion is therefore manifest-driven (with checksum verification)
    whenever a COMPOSE_MANIFEST.json is present, and otherwise restricted
    to TOP-LEVEL *.train files only.

    NEVER reintroduce a recursive ('**') glob here or anywhere downstream.

    Args:
        training_corpus_path: Directory holding the corpus *.train files.

    Returns:
        A list of absolute/normalised file paths, in a deterministic order
        (manifest order if a manifest exists, else sorted by filename).

    Raises:
        FileNotFoundError / ValueError: if the manifest references a file
            that is missing or whose streamed sha256 does not match the
            recorded ``output_checksum``. We never silently fall back to a
            glob when a manifest is present.
    """
    manifest_path = os.path.join(training_corpus_path, COMPOSE_MANIFEST_NAME)

    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        per_file = manifest.get("per_file") or []
        if not per_file:
            msg = (
                f"FATAL: {manifest_path} has no 'per_file' entries; refusing "
                f"to ingest. A compose manifest must enumerate every composed "
                f"file. Aborting rather than silently falling back to a glob."
            )
            print(msg)
            raise ValueError(msg)

        resolved = []
        total_bytes = 0
        for entry in per_file:
            stem = entry.get("stem")
            expected_checksum = entry.get("output_checksum")
            if not stem:
                msg = (
                    f"FATAL: {manifest_path} per_file entry missing 'stem': "
                    f"{entry!r}"
                )
                print(msg)
                raise ValueError(msg)
            file_path = os.path.join(training_corpus_path, f"{stem}.train")
            if not os.path.exists(file_path):
                msg = (
                    f"FATAL: manifest {manifest_path} references "
                    f"'{stem}.train' but it is missing at {file_path}. "
                    f"Refusing to ingest a partial corpus."
                )
                print(msg)
                raise FileNotFoundError(msg)
            if not expected_checksum:
                msg = (
                    f"FATAL: manifest {manifest_path} entry for '{stem}' has "
                    f"no 'output_checksum'; cannot verify integrity of "
                    f"{file_path}. Refusing to ingest unverifiable data."
                )
                print(msg)
                raise ValueError(msg)
            actual_checksum = _stream_sha256(file_path)
            if actual_checksum != expected_checksum:
                msg = (
                    f"FATAL: checksum mismatch for {file_path}\n"
                    f"  manifest output_checksum = {expected_checksum}\n"
                    f"  streamed sha256          = {actual_checksum}\n"
                    f"The composed file does not match the compose manifest. "
                    f"Refusing to ingest corrupted/regenerated data."
                )
                print(msg)
                raise ValueError(msg)
            total_bytes += os.path.getsize(file_path)
            resolved.append(file_path)

        manifest_train_tokens = (manifest.get("totals") or {}).get("train_tokens")
        print(
            f"  - manifest-driven: {len(resolved)} files, {total_bytes} bytes, "
            f"manifest train_tokens={manifest_train_tokens}"
        )
        return resolved

    # No manifest (e.g. raw corpora dirs like raw/en/train_90M/): TOP-LEVEL
    # *.train files ONLY, sorted. The recursive '**/*.train' glob that used
    # to live here caused the 2026-06-04 contamination incident by walking
    # into intermediate build subdirs — never reintroduce '**'.
    training_files = sorted(glob.glob(os.path.join(training_corpus_path, '*.train')))
    print(f"  - no manifest; top-level *.train files ({len(training_files)}):")
    for fp in training_files:
        print(f"      {fp}")
    return training_files


def _resolve_test_files(test_corpus_path: str) -> list:
    """Resolve top-level *.test files for a test corpus dir.

    TOP-LEVEL ONLY, sorted. The recursive '**/*.test' glob that used to
    live here is part of the same family as the 2026-06-04 contamination
    incident — never reintroduce '**'.
    """
    test_files = sorted(glob.glob(os.path.join(test_corpus_path, '*.test')))
    print(f"  - top-level *.test files ({len(test_files)}):")
    for fp in test_files:
        print(f"      {fp}")
    return test_files


def tokenize_dataset_from_config(config_path: str, force: bool = False):
    """
    Loads training and test corpora, tokenizes them using the experiment-specific SentencePiece
    model, and saves the tokenized datasets to disk.

    Args:
        config_path: Path to the experiment config YAML
        force: If True, re-tokenize even if a tokenized dataset already exists
    """
    # Defensive: CLI path has been observed to pass force as the string
    # "False" rather than the bool False. Coerce so idempotency works.
    if isinstance(force, str):
        force = force.strip().lower() in ("1", "true", "yes", "y")
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

    # Detect tokenizer type by artefact present on disk. SentencePiece
    # ships a binary `tokenizer.model`; WordPiece/BPE-via-HF ships
    # `tokenizer.json`. This is also what cache_keys.compute_cache_key
    # auto-detects, so the hash is stable for either.
    sp_model_path = os.path.join(tokenizer_dir, 'tokenizer.model')
    hf_model_path = os.path.join(tokenizer_dir, 'tokenizer.json')
    if os.path.exists(sp_model_path):
        tokenizer_kind = 'sentencepiece'
        tokenizer_model_path = sp_model_path
    elif os.path.exists(hf_model_path):
        tokenizer_kind = 'wordpiece'
        tokenizer_model_path = hf_model_path
    else:
        raise FileNotFoundError(
            f"No tokenizer artefact in '{tokenizer_dir}' "
            f"(expected tokenizer.model or tokenizer.json).\n"
            f"  Train it once with:\n"
            f"    kubectl apply -f k8s/job-train-tokenizer.yaml\n"
            f"  (edit TOKENIZER_CONFIG env if your config.tokenizer.output_dir "
            f"differs from the default)."
        )

    if not os.path.exists(training_corpus_path):
        raise FileNotFoundError(
            f"Training corpus path not found at '{training_corpus_path}'."
        )

    # 0.2 — content-addressed cache path so different experiments that
    # share corpus + tokenizer + seq_length + manipulation reuse tokenized
    # output. Falls back to the legacy experiment_name-keyed path when a
    # run that predates this change already populated it.
    max_sequence_length = int(config['data']['max_sequence_length'])
    manipulation = config.get('dataset_manipulation') or []
    cache_key = compute_cache_key(
        training_corpus_path, tokenizer_dir, max_sequence_length, manipulation
    )
    tokenized_data_dir = os.path.join(base_dir, "data", "tokenized", cache_key)
    legacy_tokenized_data_dir = os.path.join(base_dir, "data", "tokenized", experiment_name)

    print(f"  - Experiment:          {experiment_name}")
    print(f"  - Tokenizer Model:     {tokenizer_model_path}")
    print(f"  - Cache key:           {cache_key}")
    print(f"  - Output Directory:    {tokenized_data_dir}")

    # Idempotency check: skip if tokenized dataset already exists. The data
    # loader expects a saved HF Dataset under train/ (and optionally test/),
    # so we look for the canonical dataset_info.json sentinel — at either
    # the hashed path (new) or the legacy experiment_name path (runs that
    # predate 0.2, e.g. the in-flight baseline).
    #
    # CephFS attribute caching can briefly report fresh PVC entries as
    # absent. A forced listdir on the parent of the cache directory
    # invalidates the attr cache before we stat the sentinel.
    for parent in (os.path.dirname(tokenized_data_dir), os.path.dirname(legacy_tokenized_data_dir)):
        try:
            if os.path.isdir(parent):
                os.listdir(parent)
        except OSError:
            pass
    train_sentinel = os.path.join(tokenized_data_dir, 'train', 'dataset_info.json')
    legacy_train_sentinel = os.path.join(legacy_tokenized_data_dir, 'train', 'dataset_info.json')
    if not force and os.path.exists(train_sentinel):
        print(f"  - Tokenized dataset already exists at: {tokenized_data_dir}")
        print(f"  - Skipping re-tokenization. Pass force=True (or --force on CLI) to override.")
        print("----- Dataset Tokenization Skipped (cached) -----")
        return
    if not force:
        # Log why the cache check failed, to aid debugging.
        print(f"  - Cache miss — hashed sentinel {train_sentinel} exists={os.path.exists(train_sentinel)}, "
              f"legacy sentinel {legacy_train_sentinel} exists={os.path.exists(legacy_train_sentinel)}")
    if not force and os.path.exists(legacy_train_sentinel):
        print(f"  - Found legacy tokenized dataset at: {legacy_tokenized_data_dir}")
        print(f"  - Using legacy cache; new runs with the same inputs will reuse the hashed path {cache_key}.")
        print("----- Dataset Tokenization Skipped (legacy cache) -----")
        return

    if tokenizer_kind == 'sentencepiece':
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(tokenizer_model_path)
        vocab_size = tokenizer.vocab_size()
        def tokenize_batch(texts):
            return tokenizer.encode(texts, out_type=int)
    else:
        # WordPiece (BERT) via HuggingFace tokenizers. PreTrainedTokenizerFast
        # wraps the tokenizer.json and exposes `.encode` for single texts +
        # `.batch_encode_plus` for batches.
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_model_path)
        vocab_size = tokenizer.vocab_size
        def tokenize_batch(texts):
            enc = tokenizer(texts, add_special_tokens=False, truncation=False, padding=False)
            return enc["input_ids"]
    print(f"  - Successfully loaded {tokenizer_kind} tokenizer with vocab size: {vocab_size}")

    # Process training data
    print(f"\n  - Processing training data from '{training_corpus_path}'...")
    # Manifest-driven (with checksum verification) when a COMPOSE_MANIFEST.json
    # is present; otherwise TOP-LEVEL *.train files only. Replaces the
    # recursive '**/*.train' glob that caused the 2026-06-04 contamination
    # incident (build intermediates ingested as corpus). See helper docstring.
    training_files = _resolve_training_files(training_corpus_path)

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
        return {'input_ids': tokenize_batch(examples['text'])}

    # num_proc: respect cgroup CPU affinity (sched_getaffinity) rather than
    # host cpu_count. Under K8s limits like cpu=4, os.cpu_count() still
    # reports the host's 32+ CPUs and we end up oversubscribed 10x+,
    # causing worker processes to crash with
    # `ValueError: I/O operation on closed file` inside multiprocess.pool.
    # Cap further at 8 workers for the HF datasets map step — beyond that
    # SentencePiece encode contention and arrow shard IPC become the bottleneck.
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    num_proc = max(1, min(8, available))
    print(f"  - Tokenizing with num_proc={num_proc} "
          f"(cpu_count={os.cpu_count()}, affinity={available})")
    tokenized_training_dataset = raw_training_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=['text']
    )
    print("  - Training tokenization complete.")

    # Process test data if available
    tokenized_test_dataset = None
    if test_corpus_path_from_config:
        test_corpus_path = test_corpus_path_from_config if os.path.isabs(test_corpus_path_from_config) else os.path.join(base_dir, test_corpus_path_from_config)
        
        if os.path.exists(test_corpus_path):
            print(f"\n  - Processing test data from '{test_corpus_path}'...")
            # TOP-LEVEL *.test files only — same anti-contamination rule as
            # the training discovery; never recurse with '**'.
            test_files = _resolve_test_files(test_corpus_path)

            if test_files:
                print(f"  - Loading {len(test_files)} test file(s)...")
                raw_test_dataset = load_dataset('text', data_files={'test': test_files}, split='test', cache_dir=cache_dir)
                print(f"  - Found {len(raw_test_dataset):,} total lines in the test corpus.")

                print(f"  - Tokenizing test dataset with num_proc={num_proc}...")
                tokenized_test_dataset = raw_test_dataset.map(
                    tokenize_function,
                    batched=True,
                    num_proc=num_proc,
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

    # 0.2 — write sidecar metadata so the hashed directory is auditable
    # without recomputing the key.
    meta = cache_meta(
        training_corpus_path, tokenizer_dir, max_sequence_length, manipulation,
        extra={
            "first_experiment_name": experiment_name,
            "cache_key": cache_key,
            "stage": "tokenized",
        },
    )
    with open(os.path.join(tokenized_data_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    
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