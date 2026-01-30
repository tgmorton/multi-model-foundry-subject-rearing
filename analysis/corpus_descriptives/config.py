"""
Configuration model for corpus descriptive analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from .constants import DEFAULT_GENRE_MAP_EN


class CorpusAnalysisConfig(BaseModel):
    """Configuration for a corpus descriptive analysis run."""

    input_path: Path = Field(..., description="Corpus directory containing .train files")
    output_path: Path = Field(..., description="Results output directory")
    split_name: str = Field(..., description="Split identifier (e.g. train_90M)")

    # spaCy settings
    spacy_model: str = Field("en_core_web_trf", description="spaCy model name")
    spacy_batch_size: int = Field(50, gt=0, description="Batch size for nlp.pipe()")
    spacy_disable_components: List[str] = Field(
        default_factory=lambda: ["ner"],
        description="spaCy components to disable",
    )

    # Processing
    chunk_size: int = Field(1000, gt=0, description="Lines per processing chunk")
    language: str = Field("en", description="Language code")

    # Analysis selection
    analyses: List[str] = Field(
        default_factory=lambda: [
            "pronoun_inventory",
            "expletives",
            "verb_finiteness",
            "clause_structure",
            "wh_questions",
            "relative_clauses",
            "negation",
            "that_trace",
        ],
        description="Analyses to run (default: all 8)",
    )

    # Genre mapping
    genre_map: Dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_GENRE_MAP_EN),
        description="Filename stem → genre display name",
    )

    # Checkpointing
    checkpoint_dir: Optional[Path] = Field(None, description="Checkpoint directory")
    checkpoint_interval: int = Field(
        10000, gt=0, description="Lines between checkpoints"
    )

    @field_validator("input_path", "output_path", "checkpoint_dir", mode="before")
    @classmethod
    def resolve_paths(cls, v):
        """Resolve relative paths against project root."""
        if v is None:
            return v
        path = Path(v)
        if not path.is_absolute():
            from preprocessing.utils import find_project_root

            root = find_project_root(__file__)
            return root / path
        return path
