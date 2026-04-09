#!/usr/bin/env python3
"""Download diverse Spanish-English parallel corpora.

Produces paired .es / .en files per source in data/spanish/parallel/.
Sources cover a range of registers: parliamentary, conversational, educational,
news, literary, medical, diplomatic, encyclopedic.

Usage
-----
    python scripts/build_spanish_parallel.py download --data_root data/spanish
    python scripts/build_spanish_parallel.py stats --data_root data/spanish
"""

from __future__ import annotations

import argparse
import gzip
import io
import logging
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Parallel corpus sources ──────────────────────────────────────────
# Each entry: (name, description, OPUS download URL for Moses en-es.txt.zip)
# Moses format ZIPs contain two files: corpus.en and corpus.es (line-aligned)

OPUS_CORPORA = [
    (
        "europarl",
        "European Parliament proceedings",
        "https://object.pouta.csc.fi/OPUS-Europarl/v8/moses/en-es.txt.zip",
    ),
    (
        "opensubtitles",
        "Movie/TV subtitles (conversational)",
        "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-es.txt.zip",
    ),
    (
        "qed",
        "Educational video subtitles",
        "https://object.pouta.csc.fi/OPUS-QED/v2.0a/moses/en-es.txt.zip",
    ),
    (
        "ted2020",
        "TED talk transcripts",
        "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/en-es.txt.zip",
    ),
    (
        "tatoeba",
        "Crowd-sourced short sentences",
        "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/en-es.txt.zip",
    ),
    (
        "news_commentary",
        "News commentary / op-eds",
        "https://object.pouta.csc.fi/OPUS-News-Commentary/v16/moses/en-es.txt.zip",
    ),
    (
        "globalvoices",
        "Citizen journalism / global news",
        "https://object.pouta.csc.fi/OPUS-GlobalVoices/v2018q4/moses/en-es.txt.zip",
    ),
    (
        "wikimatrix",
        "Wikipedia-mined parallel sentences",
        "https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/moses/en-es.txt.zip",
    ),
    (
        "books",
        "Literary / copyright-free books",
        "https://object.pouta.csc.fi/OPUS-Books/v1/moses/en-es.txt.zip",
    ),
    (
        "emea",
        "Medical / pharmaceutical (EMA)",
        "https://object.pouta.csc.fi/OPUS-EMEA/v3/moses/en-es.txt.zip",
    ),
    (
        "unpc",
        "UN diplomatic / international affairs",
        "https://object.pouta.csc.fi/OPUS-UNPC/v1.0/moses/en-es.txt.zip",
    ),
]


def _download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress logging."""
    import urllib.request

    log.info("Downloading %s from %s", desc or dest.name, url)
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = dest.stat().st_size / 1024 / 1024
        log.info("  Downloaded %.1f MB → %s", size_mb, dest.name)
        return True
    except Exception as e:
        log.error("  Download failed: %s", e)
        return False


def _extract_moses_zip(zip_path: Path, out_dir: Path, corpus_name: str) -> bool:
    """Extract Moses-format ZIP (contains *.en and *.es files)."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            en_file = next((n for n in names if n.endswith(".en")), None)
            es_file = next((n for n in names if n.endswith(".es")), None)

            if not en_file or not es_file:
                log.error("  ZIP missing .en or .es file: %s", names)
                return False

            # Extract and rename
            en_out = out_dir / f"{corpus_name}.en"
            es_out = out_dir / f"{corpus_name}.es"

            for src_name, dst_path in [(en_file, en_out), (es_file, es_out)]:
                with zf.open(src_name) as src, open(dst_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

            return True
    except zipfile.BadZipFile:
        log.error("  Bad ZIP file: %s", zip_path)
        return False


def download_corpus(out_dir: Path, name: str, desc: str, url: str) -> None:
    """Download and extract a single OPUS parallel corpus."""
    en_path = out_dir / f"{name}.en"
    es_path = out_dir / f"{name}.es"

    if en_path.exists() and es_path.exists():
        log.info("Skipping %s (already exists)", name)
        return

    zip_path = out_dir / f"{name}.zip"

    if not _download_file(url, zip_path, f"{name} ({desc})"):
        return

    log.info("  Extracting %s...", name)
    if _extract_moses_zip(zip_path, out_dir, name):
        _log_parallel_stats(out_dir, name)
    else:
        # Some OPUS ZIPs contain .gz files inside
        log.info("  Trying alternative extraction...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(out_dir / f"{name}_tmp")
            # Look for extracted files
            tmp_dir = out_dir / f"{name}_tmp"
            for f in tmp_dir.rglob("*.en"):
                shutil.move(str(f), str(en_path))
            for f in tmp_dir.rglob("*.es"):
                shutil.move(str(f), str(es_path))
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if en_path.exists() and es_path.exists():
                _log_parallel_stats(out_dir, name)
            else:
                log.error("  Could not find .en/.es files in %s", name)
        except Exception as e:
            log.error("  Alternative extraction also failed: %s", e)

    zip_path.unlink(missing_ok=True)


def _log_parallel_stats(out_dir: Path, name: str) -> None:
    """Log line counts and verify alignment."""
    en_path = out_dir / f"{name}.en"
    es_path = out_dir / f"{name}.es"

    en_lines = sum(1 for _ in open(en_path, "r", encoding="utf-8"))
    es_lines = sum(1 for _ in open(es_path, "r", encoding="utf-8"))

    en_size = en_path.stat().st_size / 1024 / 1024
    es_size = es_path.stat().st_size / 1024 / 1024

    aligned = "OK" if en_lines == es_lines else "MISALIGNED!"
    log.info("  → %s: %d pairs (%s)  en=%.1fMB  es=%.1fMB",
             name, en_lines, aligned, en_size, es_size)

    if en_lines != es_lines:
        log.warning("  WARNING: %s has %d EN lines but %d ES lines!",
                     name, en_lines, es_lines)


def print_stats(out_dir: Path) -> None:
    """Print stats for all downloaded parallel corpora."""
    print(f"\n{'Source':<20} {'Pairs':>12} {'EN MB':>8} {'ES MB':>8} {'Status'}")
    print("-" * 60)

    total_pairs = 0
    for name, desc, _ in OPUS_CORPORA:
        en_path = out_dir / f"{name}.en"
        es_path = out_dir / f"{name}.es"

        if not en_path.exists() or not es_path.exists():
            print(f"{name:<20} {'--':>12} {'--':>8} {'--':>8} missing")
            continue

        en_lines = sum(1 for _ in open(en_path, "r", encoding="utf-8"))
        es_lines = sum(1 for _ in open(es_path, "r", encoding="utf-8"))
        en_mb = en_path.stat().st_size / 1024 / 1024
        es_mb = es_path.stat().st_size / 1024 / 1024

        status = "OK" if en_lines == es_lines else f"MISALIGNED ({en_lines} vs {es_lines})"
        print(f"{name:<20} {en_lines:>12,} {en_mb:>7.1f}M {es_mb:>7.1f}M {status}")
        total_pairs += min(en_lines, es_lines)

    print("-" * 60)
    print(f"{'TOTAL':<20} {total_pairs:>12,}")
    print()


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Spanish-English parallel corpora."
    )
    parser.add_argument(
        "action",
        choices=["download", "stats"],
        help="'download' fetches all corpora, 'stats' shows current inventory.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/spanish",
        help="Root of the Spanish data directory (default: data/spanish).",
    )
    parser.add_argument(
        "--corpora",
        type=str,
        nargs="*",
        help="Only download these corpora (space-separated names).",
    )
    args = parser.parse_args()

    out_dir = Path(args.data_root) / "parallel"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.action == "download":
        log.info("=" * 60)
        log.info("DOWNLOADING SPANISH-ENGLISH PARALLEL CORPORA")
        log.info("=" * 60)

        for name, desc, url in OPUS_CORPORA:
            if args.corpora and name not in args.corpora:
                continue
            download_corpus(out_dir, name, desc, url)

        log.info("=" * 60)
        log.info("Download complete.")
        log.info("=" * 60)
        print_stats(out_dir)

    elif args.action == "stats":
        print_stats(out_dir)


if __name__ == "__main__":
    main()
