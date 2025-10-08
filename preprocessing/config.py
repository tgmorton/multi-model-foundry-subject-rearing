"""
Preprocessing Configuration Models

Pydantic models for type-safe configuration of ablation pipelines.
Handles validation, path resolution, and reproducibility settings.
"""

import random
import sys
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class AblationConfig(BaseModel):
    """
    Configuration for a single ablation step.

    This model defines all parameters needed to run an ablation transformation,
    including reproducibility settings, processing parameters, and logging options.

    Example:
        >>> config = AblationConfig(
        ...     type="remove_articles",
        ...     input_path="data/raw/corpus/",
        ...     output_path="data/processed/corpus/",
        ...     seed=42
        ... )
    """

    # Core parameters
    type: str = Field(..., description="Ablation type (registered name)")
    input_path: Path = Field(..., description="Input corpus directory")
    output_path: Path = Field(..., description="Output directory for ablated corpus")

    # Reproducibility
    seed: int = Field(42, description="Random seed for reproducibility")

    # Processing parameters
    chunk_size: int = Field(1000, gt=0, description="Lines per processing chunk")
    skip_validation: bool = Field(False, description="Skip validation step to save time")

    # Replacement pool configuration
    replacement_pool_dir: Optional[Path] = Field(
        None,
        description="Directory containing replacement pool files"
    )

    # spaCy configuration
    spacy_model: str = Field(
        "en_core_web_sm",
        description="spaCy model to use (en_core_web_sm or en_core_web_trf)"
    )
    spacy_device: Optional[str] = Field(
        None,
        description="Device: 'cpu', 'cuda', 'mps', or None for auto-detect"
    )

    # Logging
    verbose: bool = Field(False, description="Enable verbose logging")
    log_dir: Path = Field(Path("logs"), description="Log output directory")

    # Additional ablation-specific parameters
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Ablation-specific parameters"
    )

    @field_validator('input_path', 'output_path', 'replacement_pool_dir', 'log_dir', mode='before')
    @classmethod
    def resolve_paths(cls, v):
        """Resolve paths relative to project root if not absolute."""
        if v is None:
            return v

        path = Path(v)
        if not path.is_absolute():
            # Import here to avoid circular dependency
            from preprocessing.utils import find_project_root
            root = find_project_root(__file__)
            return root / path
        return path

    @model_validator(mode='after')
    def set_random_seeds(self):
        """Set random seeds for reproducibility after initialization."""
        random.seed(self.seed)

        # Set numpy seed if available
        try:
            import numpy as np
            np.random.seed(self.seed)
        except ImportError:
            pass

        # Set torch seed if available
        try:
            import torch
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        except ImportError:
            pass

        return self


class ProvenanceMetadata(BaseModel):
    """
    Metadata for tracking ablation provenance.

    Captures complete environment information and processing statistics
    to enable reproducibility and result tracking.
    """

    # Execution environment
    timestamp: str = Field(..., description="ISO format timestamp")
    python_version: str = Field(..., description="Python version string")
    spacy_version: str = Field(..., description="spaCy version")
    spacy_model_name: str = Field(..., description="spaCy model name")
    spacy_model_version: str = Field(..., description="spaCy model version")
    device: str = Field(..., description="Processing device (cpu/cuda/mps)")
    hostname: str = Field(..., description="Machine hostname")
    platform: str = Field(..., description="OS platform")

    # Configuration
    ablation_type: str = Field(..., description="Type of ablation performed")
    random_seed: int = Field(..., description="Random seed used")
    chunk_size: int = Field(..., description="Processing chunk size")

    # Input/Output checksums
    input_checksums: Dict[str, str] = Field(
        default_factory=dict,
        description="SHA256 checksums of input files"
    )
    output_checksums: Dict[str, str] = Field(
        default_factory=dict,
        description="SHA256 checksums of output files"
    )

    # Statistics
    total_files_processed: int = Field(0, ge=0, description="Number of files processed")
    total_tokens_original: int = Field(0, ge=0, description="Original token count")
    total_tokens_final: int = Field(0, ge=0, description="Final token count")
    total_items_ablated: int = Field(
        0,
        ge=0,
        description="Total items ablated (expletives, articles, etc.)"
    )
    processing_time_seconds: float = Field(0.0, ge=0.0, description="Processing duration")

    @classmethod
    def create_from_environment(
        cls,
        ablation_type: str,
        random_seed: int,
        chunk_size: int,
        device: str,
        spacy_version: str,
        spacy_model_name: str,
        spacy_model_version: str
    ) -> "ProvenanceMetadata":
        """
        Create metadata from current environment.

        Args:
            ablation_type: Type of ablation being performed
            random_seed: Random seed used
            chunk_size: Processing chunk size
            device: Device being used (cpu/cuda/mps)
            spacy_version: spaCy version
            spacy_model_name: spaCy model name
            spacy_model_version: spaCy model version

        Returns:
            ProvenanceMetadata instance
        """
        return cls(
            timestamp=datetime.utcnow().isoformat() + "Z",
            python_version=sys.version,
            spacy_version=spacy_version,
            spacy_model_name=spacy_model_name,
            spacy_model_version=spacy_model_version,
            device=device,
            hostname=platform.node(),
            platform=platform.platform(),
            ablation_type=ablation_type,
            random_seed=random_seed,
            chunk_size=chunk_size
        )


class FileStatistics(BaseModel):
    """Statistics for a single processed file."""

    file_name: str = Field(..., description="Name of the file")
    original_tokens: int = Field(0, ge=0, description="Original token count")
    final_tokens: int = Field(0, ge=0, description="Final token count")
    items_ablated: int = Field(0, ge=0, description="Number of items ablated")
    proportion_removed: float = Field(0.0, ge=0.0, le=1.0, description="Proportion removed")
    processing_time_seconds: float = Field(0.0, ge=0.0, description="Processing time")


class ProvenanceManifest(BaseModel):
    """
    Complete provenance record for an ablation run.

    This manifest captures all information needed to reproduce an ablation,
    including environment details, configuration, and processing statistics.
    """

    metadata: ProvenanceMetadata
    file_statistics: List[FileStatistics] = Field(default_factory=list)
    config: Dict[str, Any] = Field(..., description="Full ablation configuration")

    def save(self, output_dir: Path) -> Path:
        """
        Save manifest to JSON file.

        Args:
            output_dir: Directory to save the manifest

        Returns:
            Path to the saved manifest file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = output_dir / "ABLATION_MANIFEST.json"
        manifest_path.write_text(self.model_dump_json(indent=2))

        return manifest_path

    def add_file_stats(self, stats: FileStatistics) -> None:
        """
        Add file statistics to the manifest.

        Args:
            stats: FileStatistics instance to add
        """
        self.file_statistics.append(stats)

        # Update aggregate statistics
        self.metadata.total_files_processed += 1
        self.metadata.total_tokens_original += stats.original_tokens
        self.metadata.total_tokens_final += stats.final_tokens
        self.metadata.total_items_ablated += stats.items_ablated
