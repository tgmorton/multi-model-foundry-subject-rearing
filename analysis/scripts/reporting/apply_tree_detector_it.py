#!/usr/bin/env python
"""
Apply the Italian tree-based null subject detector to annotated corpus output.

Reads base layer parquets (sentence_id + text), runs TreeNullSubjectDetector
on each sentence, and writes a tree_detector layer parquet joinable on
sentence_id.

Usage
-----
    PYTHONPATH=. python -m analysis.scripts.reporting.apply_tree_detector_it \
        --annotated-dir data/output/it_local_test/annotated_corpus/ \
        --model-dir data/pronoun_recovery/tree_detector/it/ \
        --classifier-type gradient_boosted
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
import spacy

from analysis.corpus_descriptives.schema import get_layer_schema
from analysis.pronoun_recovery.tree_detector.inference import TreeNullSubjectDetector

logger = logging.getLogger(__name__)


def _build_detection_row(
    sentence_id: str,
    detections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a single row for the tree_detector layer."""
    structs = [
        {
            "token_idx": d["token_idx"],
            "verb_text": d["verb_text"],
            "verb_lemma": d["verb_lemma"],
            "label": d["label"],
            "confidence": d["confidence"],
            "lexical_form": d["lexical_form"],
        }
        for d in detections
    ]
    return {
        "sentence_id": sentence_id,
        "null_subject_detections": structs,
        "n_null_subjects": len(structs),
        "has_null_subject_tree": len(structs) > 0,
    }


def process_base_file(
    base_path: Path,
    detector: TreeNullSubjectDetector,
    output_dir: Path,
    schema: pa.Schema,
    batch_size: int = 256,
) -> int:
    """Process a single base parquet file and write tree_detector layer.

    Returns:
        Number of sentences processed.
    """
    base_table = pq.read_table(base_path, columns=["sentence_id", "text"])
    sentence_ids = base_table.column("sentence_id").to_pylist()
    texts = base_table.column("text").to_pylist()

    n = len(texts)
    logger.info("Processing %s: %d sentences", base_path.name, n)

    # Process in batches via nlp.pipe
    rows: List[Dict[str, Any]] = []

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch_texts = texts[batch_start:batch_end]
        batch_ids = sentence_ids[batch_start:batch_end]

        docs = list(detector.nlp.pipe(batch_texts))

        for sid, doc in zip(batch_ids, docs):
            detections = detector.predict_doc(doc)
            rows.append(_build_detection_row(sid, detections))

    # Write output
    output_path = output_dir / base_path.name
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, output_path, compression="snappy")

    n_detected = sum(1 for r in rows if r["has_null_subject_tree"])
    logger.info(
        "  Wrote %s: %d/%d sentences with null subjects detected",
        output_path.name,
        n_detected,
        n,
    )
    return n


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Apply Italian tree detector to annotated corpus."
    )
    parser.add_argument(
        "--annotated-dir",
        type=str,
        required=True,
        help="Path to annotated corpus directory (containing base/ and layers/)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to tree detector model directory",
    )
    parser.add_argument(
        "--classifier-type",
        type=str,
        default="gradient_boosted",
        choices=["decision_tree", "gradient_boosted"],
        help="Which classifier to use (default: gradient_boosted)",
    )
    parser.add_argument(
        "--spacy-model",
        type=str,
        default="it_core_news_lg",
        help="Italian spaCy model name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for spaCy nlp.pipe (default: 256)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Process only base parquets matching this substring (e.g. 'europarl')",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    annotated_dir = Path(args.annotated_dir)
    base_dir = annotated_dir / "base"
    if not base_dir.exists():
        logger.error("Base directory not found: %s", base_dir)
        sys.exit(1)

    # Load spaCy once, pass to detector
    logger.info("Loading spaCy model: %s", args.spacy_model)
    nlp = spacy.load(args.spacy_model, disable=["ner"])

    # Load tree detector
    logger.info("Loading tree detector from: %s", args.model_dir)
    detector = TreeNullSubjectDetector(
        model_dir=Path(args.model_dir),
        classifier_type=args.classifier_type,
        nlp=nlp,
    )

    # Prepare output directory
    output_dir = annotated_dir / "layers" / "tree_detector"
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = get_layer_schema("tree_detector")

    # Process each base parquet file
    base_files = sorted(base_dir.glob("*.parquet"))
    if args.file:
        base_files = [f for f in base_files if args.file in f.stem]
    if not base_files:
        logger.error("No base parquet files found in %s (filter=%s)", base_dir, args.file)
        sys.exit(1)

    total = 0
    for base_path in base_files:
        total += process_base_file(
            base_path, detector, output_dir, schema, args.batch_size
        )

    logger.info("Done. Total sentences processed: %d", total)


if __name__ == "__main__":
    main()
