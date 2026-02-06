#!/usr/bin/env python3
"""Reorganize Italian corpus data: sentence-segment broken files, rename, and build output directories.

Produces:
    data/italian/train_100M/   — segmented + renamed from 100M/
    data/italian/test_10M/     — segmented + renamed from test_data/

After running this script, use preprocessing/00_prepare_corpus.py to split
train_100M into train_90M + pull_10M.
"""

import argparse
import logging
import os
import random
import shutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ── File rename map ────────────────────────────────────────────────────
# old_stem → new_stem  (extensions .train / .test are preserved)
RENAME_MAP = {
    "CHILDES": "childes",
    "CltA": "clta",
    "CORPUS_ISACCO": "corpus_isacco",
    "europarl-v7.it-en.it": "europarl",
    "Leipzig_Web_Public": "leipzig_web",
    "PaCCSS-IT": "paccss",
    "QCRI": "qcri",
    "SPGC": "spgc",
}

# Files that need sentence segmentation (single giant line, has punctuation)
SEGMENT_SPACY = {"europarl", "leipzig_web", "paccss", "qcri"}

# File that needs token-chunk segmentation (no punctuation)
SEGMENT_CHUNKS = {"spgc"}

# Files that are already well-formed (one utterance/sentence per line)
COPY_ONLY = {"childes", "clta", "corpus_isacco"}

SPACY_CHUNK_CHARS = 100_000   # characters per spaCy processing chunk
TOKEN_CHUNK_SIZE = 1000       # tokens per SPGC output line
RANDOM_SEED = 42


# ── Segmentation helpers ──────────────────────────────────────────────

def _segment_with_spacy(text: str, nlp) -> list[str]:
    """Split *text* into sentences using spaCy's rule-based sentencizer.

    Processes in ~100K-character chunks (split at space boundaries) to keep
    memory usage reasonable on very large files.
    """
    sentences: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + SPACY_CHUNK_CHARS, length)
        # Avoid breaking mid-word
        if end < length:
            space_idx = text.rfind(" ", start, end)
            if space_idx > start:
                end = space_idx + 1
        chunk = text[start:end]
        doc = nlp(chunk)
        for sent in doc.sents:
            line = sent.text.strip()
            if line:
                sentences.append(line)
        start = end
    return sentences


def _segment_spgc(text: str, rng: random.Random) -> list[str]:
    """Segment SPGC text into fixed-size token chunks, shuffled IID.

    - Strips ``=== ... ===`` document markers
    - Splits remaining tokens into chunks of TOKEN_CHUNK_SIZE
    - Shuffles chunks randomly
    """
    cleaned_tokens: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("===") and stripped.endswith("==="):
            continue  # skip document marker lines
        cleaned_tokens.extend(stripped.split())

    # Build chunks
    chunks: list[str] = []
    for i in range(0, len(cleaned_tokens), TOKEN_CHUNK_SIZE):
        chunk_tokens = cleaned_tokens[i : i + TOKEN_CHUNK_SIZE]
        if chunk_tokens:
            chunks.append(" ".join(chunk_tokens))

    rng.shuffle(chunks)
    return chunks


# ── Main processing ───────────────────────────────────────────────────

def _old_stem_for(filename: str) -> str | None:
    """Return the old stem from RENAME_MAP that matches *filename*, or None."""
    for old_stem in RENAME_MAP:
        if filename.startswith(old_stem + "."):
            return old_stem
    return None


def process_directory(
    src_dir: str,
    dst_dir: str,
    extension: str,
    nlp,
    rng: random.Random,
) -> None:
    """Process all corpus files in *src_dir* and write to *dst_dir*.

    Parameters
    ----------
    src_dir : str
        Source directory containing the original corpus files.
    dst_dir : str
        Destination directory for segmented/renamed files.
    extension : str
        File extension to look for (``".train"`` or ``".test"``).
    nlp : spacy.Language
        A blank Italian spaCy pipeline with the sentencizer component.
    rng : random.Random
        Seeded RNG for SPGC shuffling.
    """
    os.makedirs(dst_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(src_dir) if f.endswith(extension))
    if not files:
        log.warning("No %s files found in %s", extension, src_dir)
        return

    for filename in files:
        old_stem = _old_stem_for(filename)
        if old_stem is None:
            log.warning("Skipping unrecognised file: %s", filename)
            continue

        new_stem = RENAME_MAP[old_stem]
        new_filename = new_stem + extension
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, new_filename)

        log.info("Processing %s → %s", filename, new_filename)

        if new_stem in COPY_ONLY:
            shutil.copy2(src_path, dst_path)
            _log_stats(dst_path)
            continue

        # Read entire file
        with open(src_path, "r", encoding="utf-8") as f:
            text = f.read()

        if new_stem in SEGMENT_SPACY:
            lines = _segment_with_spacy(text, nlp)
        elif new_stem in SEGMENT_CHUNKS:
            lines = _segment_spgc(text, rng)
        else:
            log.warning("No segmentation strategy for %s — copying as-is", new_stem)
            shutil.copy2(src_path, dst_path)
            _log_stats(dst_path)
            continue

        with open(dst_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

        _log_stats(dst_path)


def _log_stats(path: str) -> None:
    """Log line count and file size for a written file."""
    size = os.path.getsize(path)
    with open(path, "r", encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)
    log.info("  → %s  lines=%d  size=%s", os.path.basename(path), n_lines, _human_size(size))


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}TB"


# ── CLI ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reorganize Italian corpus: segment, rename, build train_100M and test_10M."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/italian",
        help="Root of the Italian data directory (default: data/italian).",
    )
    args = parser.parse_args()

    data_root = args.data_root
    src_100m = os.path.join(data_root, "100M")
    src_10m = os.path.join(data_root, "10M")
    src_test = os.path.join(data_root, "test_data")
    dst_train = os.path.join(data_root, "train_100M")
    dst_test = os.path.join(data_root, "test_10M")

    for d in (src_100m, src_test):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Expected source directory not found: {d}")

    # Build spaCy sentencizer (rule-based, no model download needed)
    import spacy

    nlp = spacy.blank("it")
    nlp.add_pipe("sentencizer")
    log.info("spaCy sentencizer ready (blank Italian pipeline)")

    rng = random.Random(RANDOM_SEED)

    # 1. Build train_100M from 100M/
    log.info("=" * 60)
    log.info("Building train_100M from 100M/")
    log.info("=" * 60)
    process_directory(src_100m, dst_train, ".train", nlp, rng)

    # 2. Build test_10M from test_data/
    log.info("=" * 60)
    log.info("Building test_10M from test_data/")
    log.info("=" * 60)
    process_directory(src_test, dst_test, ".test", nlp, rng)

    log.info("=" * 60)
    log.info("Done.  Next step:")
    log.info(
        "  python preprocessing/00_prepare_corpus.py "
        "--source_dir %s --main_output_dir %s --pool_output_dir %s "
        "--pool_words_total 10000000",
        dst_train,
        os.path.join(data_root, "train_90M"),
        os.path.join(data_root, "pull_10M"),
    )


if __name__ == "__main__":
    main()
