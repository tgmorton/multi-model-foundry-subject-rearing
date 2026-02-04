"""
CLI entry point for corpus descriptive analysis.

Supports two modes:
1. Analyzer mode (default): Aggregates counts via analyzers
2. Annotation mode: Produces per-sentence Parquet files

Usage:
    # Analyzer mode
    python -m analysis.corpus_descriptives.run --config configs/corpus_analysis_train90m.yaml

    # Annotation mode
    python -m analysis.corpus_descriptives.run --config configs/corpus_analysis_train90m.yaml --annotate

    # Annotation mode (via config)
    # Set annotation_mode: true in the config YAML
"""

import argparse
import sys
from pathlib import Path

import yaml

from .config import CorpusAnalysisConfig
from .pipeline import CorpusAnalysisPipeline, CorpusAnnotationPipeline


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run corpus descriptive analysis on a BabyLM split."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Run in annotation mode (overrides config)",
    )
    parser.add_argument(
        "--layered",
        action="store_true",
        default=None,
        help="Use layered output in annotation mode",
    )
    args = parser.parse_args(argv)

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    config = CorpusAnalysisConfig(**raw)

    # Determine mode
    annotation_mode = args.annotate or config.annotation_mode

    if annotation_mode:
        # Import annotators
        from .annotators import ANNOTATOR_REGISTRY, get_default_annotators

        # Get annotators
        annotators = []
        for name in config.annotators:
            if name in ANNOTATOR_REGISTRY:
                annotators.append(ANNOTATOR_REGISTRY[name]())
            else:
                print(f"Warning: Unknown annotator '{name}', skipping", file=sys.stderr)

        if not annotators:
            annotators = get_default_annotators()

        # Determine layered setting
        layered = args.layered if args.layered is not None else config.layered_output

        # Run annotation pipeline
        pipeline = CorpusAnnotationPipeline(
            config=config,
            annotators=annotators,
            layered=layered,
        )
        metadata = pipeline.run()

        # Summary
        print(f"\nAnnotation complete for split: {config.split_name}")
        print(f"Output: {pipeline.output_dir}")
        print(f"Total sentences: {metadata.get('total_sentences', 'N/A'):,}")
        print(f"Annotators: {', '.join(a.name for a in annotators)}")

    else:
        # Run analyzer pipeline (original mode)
        pipeline = CorpusAnalysisPipeline(config)
        results = pipeline.run()

        # Summary
        print(f"\nAnalysis complete for split: {config.split_name}")
        print(f"Output: {config.output_path}")
        for name in results:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
