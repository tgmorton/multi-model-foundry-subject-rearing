#!/usr/bin/env python3
"""Download and assemble the Spanish BebéLM corpus.

Produces one .train file per source in data/spanish/raw/, then segments and
renames them into data/spanish/train_100M/ or data/spanish/train_10M/.
After running this script, use preprocessing/00_prepare_corpus.py to split
train_100M into train_90M + pull_10M.

Two corpus sizes are supported:

  100M — Full training corpus.  Rebalanced to account for unavailable COSCACH
         spoken corpus.  Shortfalls in CHILDES, grerli, vikidia, and spoken
         are redistributed proportionally to opensubtitles, qed, and
         leipzig_web (nearest spoken/conversational registers).

  10M  — Ecologically-focused smaller corpus.  Preserves all ecologically
         valid sources at their full available size (CHILDES, child_narratives,
         grerli, spoken, vikidia), then fills the remainder from the other
         sources in proportion to their 100M weights.

Usage
-----
    # Download all sources (requires internet)
    python scripts/build_spanish_corpus.py download --data_root data/spanish

    # Assemble 100M corpus (default)
    python scripts/build_spanish_corpus.py assemble --data_root data/spanish

    # Assemble 10M corpus
    python scripts/build_spanish_corpus.py assemble --data_root data/spanish --size 10M

    # Both download + assemble
    python scripts/build_spanish_corpus.py all --data_root data/spanish
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import random
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

RANDOM_SEED = 42

# ── Target word counts per source ────────────────────────────────────
# 100M targets — rebalanced after dropping COSCACH (~9M spoken shortfall)
# and accepting shortfalls in CHILDES (monolingual only), grerli, vikidia.
# Extra words redistributed to opensubtitles (+7M), qed (+2M),
# leipzig_web (+1.6M) as nearest spoken/conversational registers.
TARGETS_100M = {
    "childes": 1_700_000,         # all available (monolingual only)
    "europarl": 34_000_000,
    "opensubtitles": 27_000_000,  # +7M from spoken shortfall
    "qed": 7_000_000,             # +2M from spoken shortfall
    "gutenberg": 10_000_000,
    "leipzig_web": 16_600_000,    # +1.6M from spoken shortfall
    "vikidia": 2_200_000,         # all available
    "child_narratives": 200_000,
    "grerli": 110_000,            # all available
    "spoken": 1_100_000,          # CORLEC only, no COSCACH
}

# 10M targets — ecologically-focused corpus.
# Preserve all ecologically valid sources at full available size,
# fill remainder from other sources in proportion to their 100M weights.
_ECOLOGICAL_SOURCES = {"childes", "child_narratives", "grerli", "spoken", "vikidia"}
_ECOLOGICAL_FIXED = {
    "childes": 1_700_000,
    "child_narratives": 278_000,
    "grerli": 110_000,
    "spoken": 1_100_000,
    "vikidia": 2_200_000,
}
_FILL_SOURCES_100M = {k: v for k, v in TARGETS_100M.items() if k not in _ECOLOGICAL_SOURCES}
_FILL_TOTAL_100M = sum(_FILL_SOURCES_100M.values())
_FILL_BUDGET_10M = 10_000_000 - sum(_ECOLOGICAL_FIXED.values())  # ~4.612M

TARGETS_10M = dict(_ECOLOGICAL_FIXED)
for _src, _w100 in _FILL_SOURCES_100M.items():
    TARGETS_10M[_src] = round(_w100 / _FILL_TOTAL_100M * _FILL_BUDGET_10M)

TARGETS = {"100M": TARGETS_100M, "10M": TARGETS_10M}

# ── Spanish CHILDES monolingual corpora ──────────────────────────────
# Only monolingual Spanish corpora — no bilingual
# Keys = display name, values = (zip_name, base_url)
# Most are on git.talkbank.org/childes/data/Spanish/
# Koine, LlinasOjea, PAIDUS are on PhonBank

CHILDES_GIT_BASE = "https://git.talkbank.org/childes/data/Spanish"
PHONBANK_GIT_BASE = "https://git.talkbank.org/phon/data/Spanish"

# Names matching the HuggingFace BabyLM-community/formatted-CHILDES "spa" config.
# These are the corpus names as they appear in the data-source field.
CHILDES_MONOLINGUAL_CDS = [
    "Aguirre", "BecaCESNo", "FernAguado", "Granada", "Marrero",
    "Nieva", "Ornat", "Oviedo", "Remedi", "SerraSole", "Vila", "Vivar",
    "Spanish-Aguilar", "Spanish-Ornat", "Spanish-Sebastian",
    "CORDIS", "MOC", "PERLA",
]

# Exclude clearly bilingual corpora
CHILDES_BILINGUAL = {"Spanish-MiamiBiling", "Gildersleeve", "JacksonThal"}

# Narrative corpora (child speech, not CDS — separate category)
CHILDES_NARRATIVE_CORPORA = ["ColMex", "Hess", "Shiro"]

# School-age spoken/written
CHILDES_SCHOOL_CORPORA = ["GRERLI", "DiezItza"]


# ── CHAT file processing ─────────────────────────────────────────────

def extract_utterances_from_chat(chat_text: str, include_child: bool = True) -> list[str]:
    """Extract utterances from CHAT-format text.

    Parameters
    ----------
    chat_text : str
        Contents of a .cha file.
    include_child : bool
        If True, include child utterances (CHI). If False, only adult speech.

    Returns
    -------
    list[str]
        Cleaned utterances, one per element.
    """
    adult_codes = {"MOT", "FAT", "ADU", "INV", "OBS", "TEA", "EXP", "GRA",
                   "UNC", "SIS", "BRO", "AUN", "DAD", "MOM", "CAR", "BAB"}
    child_codes = {"CHI", "TGT"}

    utterances = []
    current_speaker = None
    current_line = ""

    for line in chat_text.splitlines():
        # Speaker tier: *SPK:\ttext
        m = re.match(r"^\*(\w+):\t(.*)$", line)
        if m:
            # Flush previous
            if current_speaker and current_line.strip():
                utterances.append(current_line.strip())
            speaker = m.group(1)
            text = m.group(2)

            # Determine if we keep this speaker
            is_adult = speaker[:3] in adult_codes or speaker in adult_codes
            is_child = speaker[:3] in child_codes or speaker in child_codes
            if is_adult or (include_child and is_child):
                current_speaker = speaker
                current_line = text
            else:
                current_speaker = None
                current_line = ""
            continue

        # Continuation line (starts with tab)
        if line.startswith("\t") and current_speaker:
            current_line += " " + line.strip()
            continue

        # Dependent tier (%mor, %gra, etc.) — skip
        if line.startswith("%"):
            continue

        # Header or other — flush
        if current_speaker and current_line.strip():
            utterances.append(current_line.strip())
        current_speaker = None
        current_line = ""

    # Flush last
    if current_speaker and current_line.strip():
        utterances.append(current_line.strip())

    # Clean utterances
    cleaned = []
    for utt in utterances:
        utt = _clean_chat_utterance(utt)
        if utt:
            cleaned.append(utt)
    return cleaned


def _clean_chat_utterance(text: str) -> str:
    """Remove CHAT annotations from an utterance, leaving readable text."""
    # Remove retracing markers [/], [//], [///]
    text = re.sub(r"\[/+\]", "", text)
    # Remove error markers [*]
    text = re.sub(r"\[\*\]", "", text)
    # Remove bracketed annotations [= text], [% text], [+ text], etc.
    text = re.sub(r"\[.*?\]", "", text)
    # Remove angle-bracket scope markers <...>
    text = re.sub(r"<([^>]*)>", r"\1", text)
    # Remove CHAT special characters: @o, @l, @c, etc.
    text = re.sub(r"@[a-z]+", "", text)
    # Remove utterance terminators
    text = re.sub(r"[.\?!]+\s*$", "", text)
    # Remove &= actions and &+ fillers
    text = re.sub(r"&[=+]\w+", "", text)
    # Remove pauses (.) (..) (...)
    text = re.sub(r"\(\.\.*\)", "", text)
    # Remove xxx, yyy, www (unintelligible)
    text = re.sub(r"\b(xxx|yyy|www)\b", "", text)
    # Remove 0* codes (actions like 0does)
    text = re.sub(r"\b0\w+", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Download functions ───────────────────────────────────────────────

def download_childes_from_huggingface(raw_dir: Path, corpus_names: list[str],
                                     output_name: str) -> None:
    """Download Spanish CHILDES via HuggingFace formatted-CHILDES dataset.

    Uses BabyLM-community/formatted-CHILDES (no auth required) which has
    2,179 Spanish transcripts with speaker turns preserved.

    Parameters
    ----------
    corpus_names : list of str
        Corpus names to include (matched against data-source field).
        Names in CHILDES_BILINGUAL are always excluded.
    output_name : str
        Stem for the output file (e.g. "childes" → "childes.train").
    """
    out_path = raw_dir / f"{output_name}.train"
    if out_path.exists() and out_path.stat().st_size > 100:
        log.info("Skipping %s (already exists with content)", out_path)
        return

    wanted = {name.lower() for name in corpus_names}

    try:
        from datasets import load_dataset
    except ImportError:
        log.error("Install datasets: pip install datasets")
        return

    log.info("Downloading Spanish CHILDES from HuggingFace formatted-CHILDES...")
    try:
        ds = load_dataset("BabyLM-community/formatted-CHILDES", "spa", split="train")
        log.info("  Got %d Spanish transcripts", len(ds))

        lines = []
        matched = 0
        skipped_bilingual = 0
        skipped_other = 0
        for row in ds:
            # Corpus name is first part of data-source (e.g. "Aguirre/12783")
            source = row.get("data-source", "")
            corpus_id = source.split("/")[0]

            # Skip bilingual corpora
            if corpus_id in CHILDES_BILINGUAL:
                skipped_bilingual += 1
                continue

            if wanted and corpus_id.lower() not in wanted:
                skipped_other += 1
                continue
            matched += 1

            # Extract clean utterances from the transcript text
            text = row.get("text", "")
            for line in text.splitlines():
                if line.startswith("*"):
                    colon_idx = line.find(":")
                    if colon_idx > 0:
                        utt = line[colon_idx + 1:].strip()
                        utt = _clean_chat_utterance(utt)
                        if utt and len(utt) > 1:
                            lines.append(utt)

        log.info("  Matched %d transcripts, skipped %d bilingual + %d other",
                 matched, skipped_bilingual, skipped_other)
        log.info("  Extracted %d utterances", len(lines))

        if lines:
            with open(out_path, "w", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
            _log_file_stats(out_path)
        else:
            log.warning("No utterances extracted for %s!", output_name)

    except Exception as e:
        log.error("formatted-CHILDES download failed: %s", e)
        log.error(
            "Manual fallback: download .cha ZIPs from TalkBank browser\n"
            "  https://sla.talkbank.org/TBB/childes/Spanish"
        )


def download_opus_corpus(raw_dir: Path, corpus_name: str, opus_name: str,
                         output_name: str) -> None:
    """Download a monolingual Spanish corpus from OPUS using opustools or wget."""
    out_path = raw_dir / f"{output_name}.train"
    if out_path.exists():
        log.info("Skipping %s (already exists)", out_path)
        return

    # Try opustools first
    try:
        log.info("Downloading %s via opus_read...", opus_name)
        result = subprocess.run(
            [
                "opus_read",
                "-d", opus_name,
                "-s", "es",
                "-t", "es",
                "-p", "raw",
                "-wm", "moses",
                "-w", str(out_path),
            ],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode == 0 and out_path.exists():
            _log_file_stats(out_path)
            return
        log.warning("opus_read failed: %s", result.stderr[:500])
    except FileNotFoundError:
        log.info("opustools not installed, trying direct download...")
    except subprocess.TimeoutExpired:
        log.warning("opus_read timed out for %s", opus_name)

    # Fallback: direct URL download for common OPUS corpora
    urls = {
        "EuroParl": "https://object.pouta.csc.fi/OPUS-Europarl/v8/mono/es.txt.gz",
        "OpenSubtitles": "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/es.txt.gz",
        "QED": "https://object.pouta.csc.fi/OPUS-QED/v2.0a/mono/es.txt.gz",
    }

    if opus_name in urls:
        url = urls[opus_name]
        gz_path = raw_dir / f"{output_name}.txt.gz"
        log.info("Downloading %s from %s", opus_name, url)
        try:
            import urllib.request
            urllib.request.urlretrieve(url, gz_path)
            # Decompress
            import gzip
            with gzip.open(gz_path, "rt", encoding="utf-8") as gz:
                with open(out_path, "w", encoding="utf-8") as f:
                    for line in gz:
                        stripped = line.strip()
                        if stripped:
                            f.write(stripped + "\n")
            gz_path.unlink()
            _log_file_stats(out_path)
        except Exception as e:
            log.error("Failed to download %s: %s", opus_name, e)
    else:
        log.error(
            "Cannot auto-download %s. Please download manually and place at %s",
            opus_name, out_path,
        )


def download_leipzig_web(raw_dir: Path) -> None:
    """Download Leipzig Spanish web corpus."""
    out_path = raw_dir / "leipzig_web.train"
    if out_path.exists():
        log.info("Skipping leipzig_web (already exists)")
        return

    # Leipzig corpora are at https://wortschatz.uni-leipzig.de/en/download/Spanish
    # Direct URL pattern for news/web corpora
    url = "https://downloads.wortschatz-leipzig.de/corpora/spa_news_2023_1M.tar.gz"
    log.info("Downloading Leipzig Spanish web corpus from %s", url)

    try:
        import urllib.request
        import tarfile
        tar_path = raw_dir / "leipzig_spa.tar.gz"
        urllib.request.urlretrieve(url, tar_path)

        with tarfile.open(tar_path, "r:gz") as tar:
            # Find the sentences file
            for member in tar.getmembers():
                if member.name.endswith("-sentences.txt"):
                    log.info("Extracting %s", member.name)
                    f = tar.extractfile(member)
                    if f:
                        lines = []
                        for line in io.TextIOWrapper(f, encoding="utf-8"):
                            # Leipzig format: ID\tsentence
                            parts = line.strip().split("\t", 1)
                            if len(parts) == 2:
                                lines.append(parts[1])
                        with open(out_path, "w", encoding="utf-8") as out:
                            for line in lines:
                                out.write(line + "\n")
                    break

        tar_path.unlink()
        _log_file_stats(out_path)

    except Exception as e:
        log.error("Failed to download Leipzig corpus: %s", e)
        log.info(
            "Manual download: go to https://wortschatz.uni-leipzig.de/en/download/Spanish\n"
            "  Download spa_news or spa_wikipedia corpora, extract the *-sentences.txt file,\n"
            "  strip the ID column, and save as %s",
            out_path,
        )


def download_gutenberg_spanish(raw_dir: Path) -> None:
    """Download Spanish texts from Project Gutenberg via HuggingFace."""
    out_path = raw_dir / "gutenberg.train"
    if out_path.exists() and out_path.stat().st_size > 1000:
        log.info("Skipping gutenberg (already exists with content)")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        log.error("Install datasets: pip install datasets")
        return

    log.info("Downloading Spanish Gutenberg from HuggingFace manu/project_gutenberg...")
    try:
        ds = load_dataset("manu/project_gutenberg", split="es")
        log.info("  Got %d Spanish books", len(ds))

        with open(out_path, "w", encoding="utf-8") as f:
            total_words = 0
            for row in ds:
                text = row.get("text", "")
                # Strip Gutenberg header/footer
                start = re.search(
                    r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
                    text, re.IGNORECASE
                )
                end = re.search(
                    r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
                    text, re.IGNORECASE
                )
                if start:
                    text = text[start.end():]
                if end:
                    text = text[:end.start()]

                # Write non-empty lines
                for line in text.splitlines():
                    line = line.strip()
                    if line:
                        f.write(line + "\n")
                        total_words += len(line.split())

        log.info("  Total words: %d", total_words)
        _log_file_stats(out_path)

    except Exception as e:
        log.error("Gutenberg download failed: %s", e)
        log.info(
            "Manual fallback: browse https://www.gutenberg.org/browse/languages/es"
        )


def download_vikidia(raw_dir: Path) -> None:
    """Download Vikidia Spanish from Kiwix ZIM dump."""
    out_path = raw_dir / "vikidia.train"
    if out_path.exists() and out_path.stat().st_size > 1000:
        log.info("Skipping vikidia (already exists with content)")
        return

    zim_url = "https://download.kiwix.org/zim/vikidia/vikidia_es_all_nopic_2025-12.zim"
    zim_path = raw_dir / "vikidia_es.zim"

    # Download ZIM file
    if not zim_path.exists():
        log.info("Downloading Vikidia ZIM from %s...", zim_url)
        try:
            import urllib.request
            urllib.request.urlretrieve(zim_url, zim_path)
        except Exception as e:
            log.error("Failed to download Vikidia ZIM: %s", e)
            return

    # Extract text from ZIM
    log.info("Extracting text from Vikidia ZIM file...")
    try:
        from libzim.reader import Archive
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            """Simple HTML tag stripper."""
            def __init__(self):
                super().__init__()
                self.parts = []
                self._skip = False
            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style", "noscript"):
                    self._skip = True
            def handle_endtag(self, tag):
                if tag in ("script", "style", "noscript"):
                    self._skip = False
            def handle_data(self, data):
                if not self._skip:
                    self.parts.append(data)
            def get_text(self):
                return " ".join(self.parts)

        zim = Archive(str(zim_path))
        with open(out_path, "w", encoding="utf-8") as f:
            n_articles = 0
            total_words = 0
            for i in range(zim.entry_count):
                entry = zim._get_entry_by_id(i)
                if entry.is_redirect:
                    continue
                try:
                    item = entry.get_item()
                    mimetype = item.mimetype
                    if "html" not in mimetype:
                        continue
                    html = bytes(item.content).decode("utf-8", errors="replace")
                    extractor = TextExtractor()
                    extractor.feed(html)
                    text = extractor.get_text()
                    for line in text.splitlines():
                        line = re.sub(r"\s+", " ", line).strip()
                        if line and len(line) > 10:
                            f.write(line + "\n")
                            total_words += len(line.split())
                    n_articles += 1
                except Exception:
                    continue

        log.info("  Extracted %d articles, %d words", n_articles, total_words)
        _log_file_stats(out_path)
        zim_path.unlink(missing_ok=True)

    except ImportError:
        log.error(
            "Install libzim: pip install libzim\n"
            "  Or manually extract text from %s using zimdump", zim_path
        )


def download_spoken(raw_dir: Path) -> None:
    """Download Spanish spoken corpora (CORLEC only)."""
    out_path = raw_dir / "spoken.train"
    if out_path.exists():
        log.info("Skipping spoken (already exists)")
        return

    log.info(
        "Spanish spoken corpus requires manual download:\n"
        "  CORLEC (~1.1M words): http://www.lllf.uam.es/ING/Corlec.html\n"
        "  Save as %s\n"
        "  Note: COSCACH was unavailable (behind registration wall).",
        out_path,
    )
    out_path.write_text("# PLACEHOLDER: Download CORLEC spoken corpus manually\n")


# ── Assembly functions ───────────────────────────────────────────────

_SENT_BOUNDARY_RE = re.compile(
    r'(?<=[.!?…])'    # lookbehind: sentence-ending punctuation
    r'(?:\s*[\"\'\)\]»])?'  # optional closing quote/bracket
    r'\s+'             # whitespace between sentences
    r'(?=[A-ZÁÉÍÓÚÑ¿¡\"\'\(\[«])'  # lookahead: uppercase / opening punct
)


def _iter_sentences(src_path: Path, needs_segmentation: bool):
    """Yield cleaned sentences from a source file.

    If needs_segmentation is True, applies regex sentence splitting to each
    line.  Otherwise yields lines as-is (one utterance per line).
    """
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if needs_segmentation:
                for sent in _SENT_BOUNDARY_RE.split(line):
                    sent = sent.strip()
                    if sent:
                        yield sent
            else:
                yield line


def subsample_file(src_path: Path, dst_path: Path, target_words: int,
                   rng: random.Random, needs_segmentation: bool = False) -> None:
    """Subsample sentences from src to reach approximately target_words.

    For large files (>2x target), uses reservoir sampling with early stopping
    to avoid reading the entire file into memory.  Streams sentences through
    _iter_sentences, which handles segmentation on the fly.
    """
    # Quick check: file size heuristic (~5 bytes/word for Spanish).
    # If the file is small enough that it's likely under target, just read it all.
    file_size = src_path.stat().st_size
    estimated_words = file_size // 5

    if estimated_words <= target_words * 1.5:
        # Small file — read everything, use all or shuffle-subsample
        lines = list(_iter_sentences(src_path, needs_segmentation))
        if not lines:
            log.warning("Empty source: %s", src_path)
            return

        total_words = sum(len(l.split()) for l in lines)
        if total_words <= target_words:
            log.info("  %s: %d words (all, under target %d)",
                     src_path.stem, total_words, target_words)
            with open(dst_path, "w", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
            return

        rng.shuffle(lines)
        selected = []
        word_count = 0
        for line in lines:
            wc = len(line.split())
            if word_count + wc > target_words * 1.01:
                break
            selected.append(line)
            word_count += wc

        log.info("  %s: %d → %d words (subsampled to target %d)",
                 src_path.stem, total_words, word_count, target_words)
        with open(dst_path, "w", encoding="utf-8") as f:
            for line in selected:
                f.write(line + "\n")
        return

    # Large file — stream with reservoir sampling.
    # Read enough to fill a buffer of ~2x target words, then sample from that.
    buffer_target = target_words * 2
    log.info("  %s: large file (%.0fMB), streaming up to ~%dM words...",
             src_path.stem, file_size / 1e6, buffer_target // 1_000_000)

    buffer = []
    buffer_words = 0
    for sent in _iter_sentences(src_path, needs_segmentation):
        wc = len(sent.split())
        buffer.append(sent)
        buffer_words += wc
        if buffer_words >= buffer_target:
            break

    if not buffer:
        log.warning("Empty source: %s", src_path)
        return

    rng.shuffle(buffer)
    selected = []
    word_count = 0
    for line in buffer:
        wc = len(line.split())
        if word_count + wc > target_words * 1.01:
            break
        selected.append(line)
        word_count += wc

    log.info("  %s: buffered %d → selected %d words (target %d)",
             src_path.stem, buffer_words, word_count, target_words)
    with open(dst_path, "w", encoding="utf-8") as f:
        for line in selected:
            f.write(line + "\n")


def assemble_corpus(data_root: Path, size: str = "100M",
                    source_dir: Path | None = None) -> None:
    """Assemble corpus into train_{size} with segmentation and subsampling.

    Parameters
    ----------
    data_root : Path
        Root of the Spanish data directory.
    size : str
        "100M" or "10M" — selects target word counts and output directory.
    source_dir : Path or None
        Directory containing source .train files.  Defaults to data_root/raw
        for 100M, but for 10M should be data_root/train_90M to ensure no
        overlap with the held-out pool.
    """
    targets = TARGETS[size]
    raw_dir = source_dir or (data_root / "raw")
    train_dir = data_root / f"train_{size}"
    train_dir.mkdir(parents=True, exist_ok=True)

    target_total = sum(targets.values())
    log.info("Assembling %s corpus (target: %d words) → %s", size, target_total, train_dir)

    rng = random.Random(RANDOM_SEED)

    # Sources that need sentence segmentation before subsampling
    # (only when reading from raw/ — train_90M is already segmented)
    needs_segmentation = {"europarl", "opensubtitles", "qed", "leipzig_web", "spoken"}
    reading_from_raw = (raw_dir == data_root / "raw")

    for source_name, target_words in targets.items():
        raw_path = raw_dir / f"{source_name}.train"
        if not raw_path.exists() or raw_path.stat().st_size < 100:
            log.warning("Skipping %s (not found or placeholder)", source_name)
            continue

        train_path = train_dir / f"{source_name}.train"
        segment = reading_from_raw and source_name in needs_segmentation
        log.info("Processing %s → %s (target: %d words, segment: %s)",
                 source_name, train_path.name, target_words, segment)

        subsample_file(raw_path, train_path, target_words, rng,
                       needs_segmentation=segment)

    # Report
    log.info("=" * 60)
    log.info("Assembly complete (%s). Corpus stats:", size)
    total = 0
    for f in sorted(train_dir.glob("*.train")):
        wc = count_words(f)
        total += wc
        log.info("  %-20s  %10d words", f.stem, wc)
    log.info("  %-20s  %10d words", "TOTAL", total)
    log.info("=" * 60)

    if size == "100M":
        log.info(
            "Next step:\n"
            "  python preprocessing/00_prepare_corpus.py \\\n"
            "    --source_dir %s \\\n"
            "    --main_output_dir %s \\\n"
            "    --pool_output_dir %s \\\n"
            "    --pool_words_total 10000000",
            train_dir,
            data_root / "train_90M",
            data_root / "pull_10M",
        )


def count_words(path: Path) -> int:
    """Count words in a text file."""
    with open(path, "r", encoding="utf-8") as f:
        return sum(len(line.split()) for line in f)


def _log_file_stats(path: Path) -> None:
    """Log line count, word count, and file size."""
    size = path.stat().st_size
    with open(path, "r", encoding="utf-8") as f:
        lines = 0
        words = 0
        for line in f:
            lines += 1
            words += len(line.split())
    units = ["B", "KB", "MB", "GB"]
    s = size
    for u in units:
        if s < 1024:
            break
        s /= 1024
    log.info("  → %s  lines=%d  words=%d  size=%.1f%s", path.name, lines, words, s, u)


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and assemble the Spanish BebéLM corpus."
    )
    parser.add_argument(
        "action",
        choices=["download", "assemble", "all"],
        help="'download' fetches sources, 'assemble' builds corpus, 'all' does both.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/spanish",
        help="Root of the Spanish data directory (default: data/spanish).",
    )
    parser.add_argument(
        "--size",
        choices=["100M", "10M"],
        default="100M",
        help="Corpus size to assemble: '100M' (default) or '10M' (ecologically-focused).",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    raw_dir = data_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.action in ("download", "all"):
        log.info("=" * 60)
        log.info("DOWNLOADING SPANISH CORPUS SOURCES")
        log.info("=" * 60)

        # 1. CHILDES monolingual CDS
        download_childes_from_huggingface(raw_dir, CHILDES_MONOLINGUAL_CDS, "childes")

        # 2. CHILDES narrative corpora (child speech)
        download_childes_from_huggingface(raw_dir, CHILDES_NARRATIVE_CORPORA, "child_narratives")

        # 3. CHILDES school-age
        download_childes_from_huggingface(raw_dir, CHILDES_SCHOOL_CORPORA, "grerli")

        # 4. EuroParl Spanish
        download_opus_corpus(raw_dir, "EuroParl", "EuroParl", "europarl")

        # 5. OpenSubtitles Spanish
        download_opus_corpus(raw_dir, "OpenSubtitles", "OpenSubtitles", "opensubtitles")

        # 6. QED educational subtitles
        download_opus_corpus(raw_dir, "QED", "QED", "qed")

        # 7. Leipzig Web
        download_leipzig_web(raw_dir)

        # 8. Gutenberg
        download_gutenberg_spanish(raw_dir)

        # 9. Vikidia
        download_vikidia(raw_dir)

        # 10. Spoken (CORLEC only)
        download_spoken(raw_dir)

        log.info("=" * 60)
        log.info("Download phase complete. Check logs for manual steps needed.")
        log.info("=" * 60)

    if args.action in ("assemble", "all"):
        log.info("=" * 60)
        log.info("ASSEMBLING SPANISH CORPUS (%s)", args.size)
        log.info("=" * 60)
        if args.size == "10M":
            # 10M is a stratified subsample of train_90M (not raw),
            # ensuring no overlap with the held-out pull_10M pool.
            src_dir = data_root / "train_90M"
            if not src_dir.exists() or not any(src_dir.glob("*.train")):
                log.error(
                    "train_90M not found.  Run the 100M assembly + 90/10 split first:\n"
                    "  python scripts/build_spanish_corpus.py assemble --data_root %s --size 100M\n"
                    "  python preprocessing/00_prepare_corpus.py ...",
                    data_root,
                )
                return
            assemble_corpus(data_root, size="10M", source_dir=src_dir)
        else:
            assemble_corpus(data_root, size=args.size)


if __name__ == "__main__":
    main()
