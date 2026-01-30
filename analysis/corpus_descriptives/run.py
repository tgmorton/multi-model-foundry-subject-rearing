"""
CLI entry point for corpus descriptive analysis.

Usage:
    python -m analysis.corpus_descriptives.run --config configs/corpus_analysis_train90m.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml

from .config import CorpusAnalysisConfig
from .pipeline import CorpusAnalysisPipeline


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run Phase 1 corpus descriptive analysis on a BabyLM split."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
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

    # Run pipeline
    pipeline = CorpusAnalysisPipeline(config)
    results = pipeline.run()

    # Summary
    print(f"\nAnalysis complete for split: {config.split_name}")
    print(f"Output: {config.output_path}")
    for name in results:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
