"""
Preprocessing Package

Unified ablation pipeline for linguistic transformations on text corpora.

This package provides a config-driven, reproducible framework for applying
ablations (removals, replacements, transformations) to training data.

Main Components:
    - AblationPipeline: Base class for running ablations
    - AblationRegistry: Central registry for ablation functions
    - AblationConfig: Configuration model for ablations
    - ProvenanceManifest: Complete provenance tracking

Example:
    >>> from preprocessing import AblationPipeline, AblationConfig
    >>> config = AblationConfig(
    ...     type="remove_articles",
    ...     input_path="data/raw/",
    ...     output_path="data/processed/",
    ...     seed=42
    ... )
    >>> pipeline = AblationPipeline(config)
    >>> manifest = pipeline.process_corpus()
"""

from .base import AblationPipeline
from .config import (
    AblationConfig,
    FileStatistics,
    ProvenanceManifest,
    ProvenanceMetadata
)
from .registry import AblationRegistry
from .utils import (
    compute_file_checksum,
    count_tokens,
    ensure_directory_exists,
    get_environment_info,
    get_spacy_device
)

__all__ = [
    # Core classes
    "AblationPipeline",
    "AblationRegistry",

    # Configuration
    "AblationConfig",
    "FileStatistics",
    "ProvenanceManifest",
    "ProvenanceMetadata",

    # Utilities
    "compute_file_checksum",
    "count_tokens",
    "ensure_directory_exists",
    "get_environment_info",
    "get_spacy_device",
]

__version__ = "1.0.0"
