"""
Corpus annotation cache for ablation pipelines.

Parses a corpus once with spaCy and serializes every line's Doc to a
spaCy DocBin, plus a per-file line map that records how many lines
became Docs (and which lines did not). Downstream ablation runs
consume the cache instead of re-parsing — a single annotation pass
amortizes across N ablation experiments.

Output layout per corpus::

    {output_dir}/
        ANNOTATION_MANIFEST.json     # model, checksums, counts, timing
        vocab/                        # spaCy model vocab (for DocBin load)
        {genre_stem}.spacy            # DocBin of cleaned Docs
        {genre_stem}.linemap.jsonl    # one JSON per input line

Each ``linemap.jsonl`` row::

    {"line_idx": N, "doc_idx": M|null, "raw_text": "..."}

- ``doc_idx`` is non-null when the line produced a Doc in the DocBin.
- ``doc_idx`` is null for boundary markers / empty lines.
- ``raw_text`` is preserved verbatim so ablation output can pass
  untreated lines through unchanged.

Consumer contract (see ``AblationPipeline``)::

    - Open ``{genre_stem}.spacy`` with the vocab stored under ``vocab/``.
    - Iterate ``{genre_stem}.linemap.jsonl`` in order; for each entry:
        - doc_idx is null  -> write ``raw_text`` through as-is.
        - doc_idx is N     -> read DocBin[N], apply ablation, write result.

Cleaners (from ``analysis.corpus_descriptives.line_cleaners``) are
intentionally NOT applied here in this first pass — ablations retain
their current behavior of operating on raw text. A future unification
with ``CorpusAnnotationPipeline`` can add a ``--apply-cleaners`` flag
so the cache serves both ablation and Parquet analytics from a single
parse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

from .utils import (
    compute_file_checksum,
    ensure_directory_exists,
    get_spacy_device,
    setup_logging,
)


# Default root for the annotation cache. Production pods mount the
# ``corpus-analysis-data`` PVC at ``/mnt/data``; this picks up the
# canonical ``/mnt/data/annotated/`` location. Override for local dev::
#
#     export ANNOTATION_CACHE_ROOT=/tmp/annotated
#
ANNOTATION_CACHE_ROOT_ENV = "ANNOTATION_CACHE_ROOT"
_DEFAULT_ANNOTATION_CACHE_ROOT = "/mnt/data/annotated"


def compute_annotation_cache_key(
    input_dir: Path,
    spacy_model: str,
    docbin_attrs: List[str],
) -> str:
    """Deterministic 16-hex-char key for an annotation run.

    Hash ingredients (mirrors ``model_foundry/cache_keys.py`` convention):
      - abspath of ``input_dir`` (identity of the corpus — we don't hash
        file contents because a 90M-word corpus makes that too slow for
        a "what's my output dir?" question; callers are responsible for
        regenerating when source files change in place)
      - spaCy model name (different models produce different parses)
      - ``docbin_attrs`` sorted (changing attrs invalidates downstream)

    Returns:
        First 16 hex chars of SHA-256 of the canonical JSON blob.
    """
    blob = json.dumps(
        {
            "input_dir": str(Path(input_dir).resolve()),
            "spacy_model": spacy_model,
            "docbin_attrs": sorted(docbin_attrs),
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def default_annotation_output_dir(
    input_dir: Path,
    spacy_model: str,
) -> Path:
    """Compute the canonical ``/mnt/data/annotated/<key>/`` output dir.

    Respects the ``ANNOTATION_CACHE_ROOT`` env var for local dev.
    The returned directory is NOT created here — that's the caller's
    job (``annotate_corpus`` creates it).
    """
    root = Path(os.environ.get(ANNOTATION_CACHE_ROOT_ENV, _DEFAULT_ANNOTATION_CACHE_ROOT))
    key = compute_annotation_cache_key(input_dir, spacy_model, DOCBIN_ATTRS)
    return root / key


# DocBin attrs we store. Covers every attribute any current ablation reads.
# HEAD is critical — without it the dependency-tree topology is lost and
# any ablation that walks `tok.children` / `tok.head` (which is nearly all
# of them) gets wrong answers. Verified 2026-04-21: omitting HEAD caused
# `remove_expletive_sentences_es` to miss `parece que ...` because the
# ccomp child was no longer reachable from the parent verb.
DOCBIN_ATTRS = [
    "POS",
    "TAG",
    "DEP",
    "HEAD",
    "LEMMA",
    "MORPH",
    "ENT_IOB",
    "ENT_TYPE",
]


@dataclass
class _FileAnnotationStats:
    """Per-file accounting for the manifest."""

    file_name: str
    n_lines_total: int = 0
    n_docs_stored: int = 0
    n_lines_passthrough: int = 0
    input_checksum: str = ""
    docbin_bytes: int = 0
    processing_time_seconds: float = 0.0


class DocBinCollector:
    """Accumulate Docs + line-map entries for one source file.

    Both ``preprocessing.annotate`` (standalone) and
    ``analysis.corpus_descriptives.CorpusAnnotationPipeline`` (unified)
    use this collector so their DocBin outputs are byte-compatible with
    the ``AblationPipeline`` consumer contract:

    - ``{stem}.spacy``            — DocBin binary
    - ``{stem}.linemap.jsonl``    — one JSON per INPUT line with
                                     ``{line_idx, doc_idx|null, raw_text}``

    Typical use::

        collector = DocBinCollector(output_dir, stem="childes")
        for line_idx, raw_line in enumerate(lines):
            if is_passthrough(raw_line):
                collector.add_passthrough(line_idx, raw_line)
            else:
                doc = nlp(cleaned_text)
                collector.add_doc(line_idx, raw_line, doc)
        stats = collector.flush()
    """

    def __init__(self, output_dir: Path, file_stem: str) -> None:
        self.output_dir = Path(output_dir)
        self.file_stem = file_stem
        self._docbin = DocBin(attrs=DOCBIN_ATTRS, store_user_data=False)
        self._linemap_entries: List[Dict[str, Any]] = []
        self._next_doc_idx = 0

    def add_doc(self, line_idx: int, raw_text: str, doc) -> int:
        """Record a parsed Doc. Return the doc_idx assigned to it."""
        doc_idx = self._next_doc_idx
        self._docbin.add(doc)
        self._linemap_entries.append(
            {"line_idx": line_idx, "doc_idx": doc_idx, "raw_text": raw_text}
        )
        self._next_doc_idx += 1
        return doc_idx

    def add_passthrough(self, line_idx: int, raw_text: str) -> None:
        """Record an input line that produced no Doc (empty / boundary)."""
        self._linemap_entries.append(
            {"line_idx": line_idx, "doc_idx": None, "raw_text": raw_text}
        )

    @property
    def n_docs(self) -> int:
        return self._next_doc_idx

    @property
    def n_lines(self) -> int:
        return len(self._linemap_entries)

    @property
    def n_passthrough(self) -> int:
        return sum(1 for e in self._linemap_entries if e["doc_idx"] is None)

    def flush(self) -> Dict[str, Any]:
        """Write DocBin + linemap to disk. Return stats dict.

        The linemap is sorted by ``line_idx`` before writing so callers
        can interleave ``add_passthrough`` and ``add_doc`` in any order —
        the on-disk file is always in source-line order, which is what
        the ``AblationPipeline`` consumer expects.
        """
        ensure_directory_exists(self.output_dir)
        docbin_path = self.output_dir / f"{self.file_stem}.spacy"
        linemap_path = self.output_dir / f"{self.file_stem}.linemap.jsonl"

        docbin_bytes = self._docbin.to_bytes()
        docbin_path.write_bytes(docbin_bytes)

        sorted_entries = sorted(
            self._linemap_entries, key=lambda e: e["line_idx"]
        )
        with open(linemap_path, "w", encoding="utf-8") as f:
            for entry in sorted_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return {
            "docbin_bytes": len(docbin_bytes),
            "n_docs": self.n_docs,
            "n_lines": self.n_lines,
            "n_passthrough": self.n_passthrough,
        }


def dump_vocab(nlp: "spacy.Language", output_dir: Path) -> Path:
    """Dump the spaCy vocab to ``output_dir/vocab`` and return the path.

    Called once per annotation run so DocBin consumers can reconstruct
    the vocab without reloading the original spaCy model.
    """
    vocab_path = Path(output_dir) / "vocab"
    ensure_directory_exists(vocab_path)
    nlp.vocab.to_disk(vocab_path)
    return vocab_path


def write_manifest(
    output_dir: Path,
    *,
    spacy_model: str,
    spacy_model_version: str,
    spacy_version: str,
    device: str,
    files: List[Dict[str, Any]],
    processing_time_seconds: float,
    source: str = "preprocessing.annotate",
) -> Path:
    """Write ``ANNOTATION_MANIFEST.json`` at the top of the DocBin output dir.

    Args:
        output_dir: Directory containing the ``.spacy`` files + vocab/.
        spacy_model: Model name (e.g. ``es_core_news_trf``).
        spacy_model_version: Model version string.
        spacy_version: Library version.
        device: ``cpu`` / ``cuda`` / ``mps``.
        files: List of per-file stats dicts (file_name, n_lines_total,
            n_docs_stored, n_lines_passthrough, input_checksum,
            docbin_bytes, processing_time_seconds).
        processing_time_seconds: Total wall time.
        source: Label for the producer (e.g. "corpus_descriptives.unified").
    """
    manifest = {
        "source": source,
        "spacy_model": spacy_model,
        "spacy_model_version": spacy_model_version,
        "spacy_version": spacy_version,
        "docbin_attrs": list(DOCBIN_ATTRS),
        "device": device,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_lines": sum(f.get("n_lines_total", 0) for f in files),
        "total_docs": sum(f.get("n_docs_stored", 0) for f in files),
        "total_passthrough": sum(f.get("n_lines_passthrough", 0) for f in files),
        "processing_time_seconds": processing_time_seconds,
        "files": files,
        "vocab_path": "vocab",
    }
    manifest_path = Path(output_dir) / "ANNOTATION_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest_path


@dataclass
class _AnnotationManifest:
    """Manifest written to ``ANNOTATION_MANIFEST.json``."""

    spacy_model: str
    spacy_model_version: str
    spacy_version: str
    docbin_attrs: List[str]
    device: str
    timestamp: str
    total_lines: int = 0
    total_docs: int = 0
    total_passthrough: int = 0
    processing_time_seconds: float = 0.0
    files: List[Dict[str, Any]] = field(default_factory=list)
    # Optional: path to the vocab directory stored alongside.
    vocab_path: str = "vocab"


def is_passthrough_line(text: str) -> bool:
    """Return True for lines that should skip parsing but still be emitted.

    Currently: blank lines and document boundary markers of the form
    ``= = = ... = = =`` (any bracketing used by English/Italian/Spanish
    corpora). Boundary markers need to flow into the ablation output
    without being mangled by spaCy parsing of pseudo-text.
    """
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.startswith("= =") and stripped.endswith("= ="):
        return True
    return False


# Backwards-compat alias.
_is_passthrough_line = is_passthrough_line


def annotate_corpus(
    input_dir: Path,
    spacy_model: str,
    output_dir: Path,
    *,
    chunk_size: int = 1000,
    spacy_batch_size: int = 50,
    file_filter: Optional[str] = None,
    spacy_disable_components: Optional[List[str]] = None,
    verbose: bool = False,
) -> _AnnotationManifest:
    """Parse every ``.train`` file in *input_dir* to DocBin + line map.

    Args:
        input_dir: Directory containing ``*.train`` corpus files.
        spacy_model: spaCy model name (e.g. ``es_core_news_trf``).
        output_dir: Output directory. Will contain DocBin files, line
            maps, a vocab dump, and the manifest.
        chunk_size: Lines read per chunk before piping to spaCy.
        spacy_batch_size: Passed to ``nlp.pipe(batch_size=...)``.
        file_filter: Optional filename stem filter (e.g. ``"childes"``).
        spacy_disable_components: Components to disable (e.g. ``["ner"]``).
        verbose: Extra logging.

    Returns:
        The manifest describing what was written.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    ensure_directory_exists(output_dir)

    logger = setup_logging(
        name="preprocessing.annotate",
        experiment="annotate",
        phase="parse",
        log_dir=str(output_dir / "logs"),
        level=logging.DEBUG if verbose else logging.INFO,
    )

    # Device + model load
    device = get_spacy_device(verbose=verbose)
    if device != "cpu":
        spacy.prefer_gpu()
    logger.info("Loading spaCy model: %s (device=%s)", spacy_model, device)
    nlp = spacy.load(spacy_model)
    nlp.max_length = 3_000_000

    if spacy_disable_components:
        enabled = [name for name, _ in nlp.pipeline]
        to_disable = [c for c in spacy_disable_components if c in enabled]
        if to_disable:
            nlp.disable_pipes(*to_disable)
            logger.info("Disabled spaCy components: %s", to_disable)

    # Serialize vocab for later DocBin deserialization. Downstream readers
    # load `spacy.blank()` and call `nlp.vocab.from_disk(vocab_path)`, or
    # load the same model and trust that vocab matches.
    dump_vocab(nlp, output_dir)

    # Manifest header
    spacy_model_version = nlp.meta.get("version", "unknown")
    manifest = _AnnotationManifest(
        spacy_model=spacy_model,
        spacy_model_version=spacy_model_version,
        spacy_version=spacy.__version__,
        docbin_attrs=list(DOCBIN_ATTRS),
        device=device,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    # Find files
    train_files = sorted(input_dir.glob("*.train"))
    if file_filter:
        train_files = [f for f in train_files if f.stem == file_filter]
    if not train_files:
        raise FileNotFoundError(
            f"No .train files in {input_dir}"
            + (f" matching filter={file_filter!r}" if file_filter else "")
        )

    start = time.time()
    for source_path in tqdm(train_files, desc="Annotating files"):
        file_start = time.time()
        stats = _annotate_file(
            nlp=nlp,
            source_path=source_path,
            output_dir=output_dir,
            chunk_size=chunk_size,
            spacy_batch_size=spacy_batch_size,
            logger=logger,
        )
        stats.processing_time_seconds = time.time() - file_start
        manifest.files.append(stats.__dict__)
        manifest.total_lines += stats.n_lines_total
        manifest.total_docs += stats.n_docs_stored
        manifest.total_passthrough += stats.n_lines_passthrough
        logger.info(
            "  %s: %d lines -> %d docs + %d pass-through in %.1fs",
            source_path.name,
            stats.n_lines_total,
            stats.n_docs_stored,
            stats.n_lines_passthrough,
            stats.processing_time_seconds,
        )

    manifest.processing_time_seconds = time.time() - start

    manifest_path = output_dir / "ANNOTATION_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest.__dict__, indent=2, default=str))
    logger.info("Wrote manifest: %s", manifest_path)

    return manifest


def _annotate_file(
    nlp: spacy.Language,
    source_path: Path,
    output_dir: Path,
    chunk_size: int,
    spacy_batch_size: int,
    logger: logging.Logger,
) -> _FileAnnotationStats:
    """Parse one .train file, writing DocBin + line map.

    Passthrough detection uses :func:`_is_passthrough_line` — blank
    lines and ``= = = ... = = =`` markers skip parsing. Ablations
    later reinstate these verbatim via the linemap's ``raw_text`` field.
    """
    stats = _FileAnnotationStats(
        file_name=source_path.name,
        input_checksum=compute_file_checksum(source_path),
    )

    with open(source_path, "r", encoding="utf-8") as src:
        raw_lines = src.readlines()

    stats.n_lines_total = len(raw_lines)

    collector = DocBinCollector(output_dir, source_path.stem)

    # Separate parseable lines from passthrough so nlp.pipe sees a
    # contiguous batch; we replay the result into the collector in
    # original order. Trailing newlines are stripped before parsing so
    # the lemmatizer produces clean lemmas (Spanish lg model returns
    # surface form as lemma for text ending in "\n" on some verbs).
    # The raw line (including "\n") is still what ends up in the linemap
    # so ablation pass-through / re-attachment works correctly.
    parse_texts: List[str] = []
    parse_line_indices: List[int] = []
    parse_raw_lines: List[str] = []
    for line_idx, raw in enumerate(raw_lines):
        if _is_passthrough_line(raw):
            collector.add_passthrough(line_idx, raw)
        else:
            parse_texts.append(raw.rstrip("\n\r"))
            parse_raw_lines.append(raw)
            parse_line_indices.append(line_idx)

    # Stream parses in chunks, mapping each Doc back to its source line.
    doc_cursor = 0
    with tqdm(
        total=len(parse_texts), desc=f"  {source_path.name}", leave=False,
    ) as pbar:
        for chunk_start in range(0, len(parse_texts), chunk_size):
            chunk = parse_texts[chunk_start : chunk_start + chunk_size]
            for doc in nlp.pipe(chunk, batch_size=spacy_batch_size):
                line_idx = parse_line_indices[doc_cursor]
                # raw_text in the linemap is the ORIGINAL line (with "\n");
                # ablation writers use it for trailing-whitespace restore.
                collector.add_doc(line_idx, parse_raw_lines[doc_cursor], doc)
                doc_cursor += 1
                pbar.update(1)

    flushed = collector.flush()
    stats.n_docs_stored = flushed["n_docs"]
    stats.n_lines_passthrough = flushed["n_passthrough"]
    stats.docbin_bytes = flushed["docbin_bytes"]
    return stats


def load_annotated_file(
    annotated_dir: Path,
    file_stem: str,
    vocab: Optional[spacy.vocab.Vocab] = None,
) -> tuple:
    """Load a single annotated file: ``(docs, linemap)``.

    ``docs`` is a list of ``Doc`` in DocBin order.
    ``linemap`` is a list of dicts (same ordering as source lines).

    .. warning::
        Materializes every ``Doc`` in memory. For large corpora (e.g.,
        Spanish ``europarl.train`` — 1.24M docs) this can consume 5-10 GB
        of RAM just for the Doc objects. Prefer :func:`iter_annotated_file`
        for streaming access.
    """
    annotated_dir = Path(annotated_dir)
    docbin_path = annotated_dir / f"{file_stem}.spacy"
    linemap_path = annotated_dir / f"{file_stem}.linemap.jsonl"

    if not docbin_path.exists():
        raise FileNotFoundError(f"Missing DocBin for {file_stem}: {docbin_path}")
    if not linemap_path.exists():
        raise FileNotFoundError(f"Missing line map for {file_stem}: {linemap_path}")

    if vocab is None:
        vocab_path = annotated_dir / "vocab"
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"No vocab dir at {vocab_path}. Load the source spaCy model "
                "explicitly and pass `vocab=nlp.vocab`."
            )
        vocab = spacy.blank("xx").vocab
        vocab.from_disk(vocab_path)

    docbin = DocBin().from_bytes(docbin_path.read_bytes())
    docs = list(docbin.get_docs(vocab))

    linemap: List[Dict[str, Any]] = []
    with open(linemap_path, "r", encoding="utf-8") as f:
        for line in f:
            linemap.append(json.loads(line))

    return docs, linemap


def iter_annotated_file(
    annotated_dir: Path,
    file_stem: str,
    vocab: Optional[spacy.vocab.Vocab] = None,
):
    """Stream ``(entry, doc_or_none)`` pairs for one annotated file.

    Iterates the line map in source-line order and pulls Docs from the
    DocBin generator as we hit content lines. Memory is O(1) in the
    number of docs (vs. :func:`load_annotated_file` which materializes
    every Doc upfront — prohibitive for large corpora like Spanish
    ``europarl.train`` at 1.24M docs / ~8 GB of Doc objects).

    Assumes the line map's ``doc_idx`` values are monotonically increasing
    across content lines — true by construction of
    :class:`DocBinCollector` (``_next_doc_idx`` increments once per
    ``add_doc`` call, in the same order docs are stored in the DocBin).

    Yields:
        Tuples of ``(linemap_entry_dict, Doc | None)``. The Doc is ``None``
        for pass-through lines (empty / boundary markers); otherwise it's
        the deserialized spaCy Doc for that source line.
    """
    annotated_dir = Path(annotated_dir)
    docbin_path = annotated_dir / f"{file_stem}.spacy"
    linemap_path = annotated_dir / f"{file_stem}.linemap.jsonl"

    if not docbin_path.exists():
        raise FileNotFoundError(f"Missing DocBin for {file_stem}: {docbin_path}")
    if not linemap_path.exists():
        raise FileNotFoundError(f"Missing line map for {file_stem}: {linemap_path}")

    if vocab is None:
        vocab_path = annotated_dir / "vocab"
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"No vocab dir at {vocab_path}. Load the source spaCy model "
                "explicitly and pass `vocab=nlp.vocab`."
            )
        vocab = spacy.blank("xx").vocab
        vocab.from_disk(vocab_path)

    docbin = DocBin().from_bytes(docbin_path.read_bytes())
    doc_iter = docbin.get_docs(vocab)
    next_expected_doc_idx = 0

    with open(linemap_path, "r", encoding="utf-8") as f:
        for raw in f:
            entry = json.loads(raw)
            doc_idx = entry.get("doc_idx")
            if doc_idx is None:
                yield entry, None
                continue
            # Guard against misordered linemap (shouldn't happen with
            # DocBinCollector; the assumption is explicit in the docstring).
            if doc_idx != next_expected_doc_idx:
                raise RuntimeError(
                    f"DocBin streaming invariant broken in {file_stem}: "
                    f"expected doc_idx={next_expected_doc_idx} but linemap "
                    f"entry has doc_idx={doc_idx}. Fall back to "
                    "load_annotated_file for random access."
                )
            try:
                doc = next(doc_iter)
            except StopIteration:
                raise RuntimeError(
                    f"DocBin in {file_stem} exhausted at doc_idx={doc_idx} "
                    "but linemap still references more docs."
                )
            next_expected_doc_idx += 1
            yield entry, doc


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m preprocessing.annotate",
        description="Annotate a corpus once; reuse for every ablation.",
    )
    p.add_argument("--input", type=Path, required=True, help="Corpus dir with .train files")
    p.add_argument(
        "--output", type=Path, default=None,
        help=(
            "Output directory. If omitted, defaults to a content-addressed path "
            "under $ANNOTATION_CACHE_ROOT (default /mnt/data/annotated/<hash>/), "
            "where <hash> is derived from (input_dir, spacy_model, docbin_attrs). "
            "Lets two pipelines asking for the same annotation share the cache."
        ),
    )
    p.add_argument("--spacy-model", type=str, required=True, help="spaCy model name")
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--spacy-batch-size", type=int, default=50)
    p.add_argument("--file", type=str, default=None, help="Process only this stem")
    p.add_argument(
        "--disable",
        nargs="*",
        default=None,
        help="spaCy components to disable (e.g. ner textcat)",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    # Resolve output dir: explicit --output wins; otherwise content-addressed
    # default under $ANNOTATION_CACHE_ROOT (see module constants).
    output_dir = args.output
    if output_dir is None:
        output_dir = default_annotation_output_dir(args.input, args.spacy_model)
        print(
            f"[annotate] --output not given; using content-addressed default: "
            f"{output_dir}",
            file=sys.stderr,
        )

    manifest = annotate_corpus(
        input_dir=args.input,
        spacy_model=args.spacy_model,
        output_dir=output_dir,
        chunk_size=args.chunk_size,
        spacy_batch_size=args.spacy_batch_size,
        file_filter=args.file,
        spacy_disable_components=args.disable,
        verbose=args.verbose,
    )
    print(
        f"Annotated {manifest.total_docs:,} docs from "
        f"{len(manifest.files)} files in {manifest.processing_time_seconds:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
