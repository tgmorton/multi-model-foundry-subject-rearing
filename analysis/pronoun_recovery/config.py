"""
Pydantic configuration models for all pronoun recovery pipeline steps.

Follows the pattern in analysis/corpus_descriptives/config.py.
"""

from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .constants import DEFAULT_GENRE_MAP_EN, DEFAULT_GENRE_MAP_IT

# ── Language-aware default lookups ────────────────────────────────────
_GENRE_MAPS = {"en": DEFAULT_GENRE_MAP_EN, "it": DEFAULT_GENRE_MAP_IT}
_SPACY_MODELS = {"en": "en_core_web_trf", "it": "it_core_news_lg"}


class _BaseConfig(BaseModel):
    """Base config that disables protected namespace warnings for model_* fields."""
    model_config = ConfigDict(protected_namespaces=())


def _resolve_path(v):
    """Resolve relative paths against project root."""
    if v is None:
        return v
    path = Path(v)
    if not path.is_absolute():
        from preprocessing.utils import find_project_root
        root = find_project_root(__file__)
        return root / path
    return path


def _apply_language_defaults(cfg):
    """Swap English-default genre_map / spacy_model when language != 'en'.

    Only triggers when the field still holds the English default value,
    so explicit YAML overrides are never clobbered.
    """
    lang = cfg.language
    if lang == "en":
        return cfg
    if hasattr(cfg, "genre_map") and cfg.genre_map == DEFAULT_GENRE_MAP_EN:
        cfg.genre_map = dict(_GENRE_MAPS.get(lang, DEFAULT_GENRE_MAP_EN))
    if hasattr(cfg, "spacy_model") and cfg.spacy_model == "en_core_web_trf":
        cfg.spacy_model = _SPACY_MODELS.get(lang, cfg.spacy_model)
    return cfg


class SyntheticDataConfig(_BaseConfig):
    """Configuration for Step 1: synthetic pair generation."""

    input_path: Path = Field(..., description="Corpus directory with .train files")
    output_path: Path = Field(
        "data/pronoun_recovery/synthetic/en",
        description="Output directory for JSONL pairs",
    )
    language: str = Field("en", description="Language code")

    # spaCy
    spacy_model: str = Field("en_core_web_trf", description="spaCy model name")
    spacy_batch_size: int = Field(50, gt=0)

    # Filtering
    min_tokens: int = Field(3, ge=1, description="Min tokens per line")
    max_tokens: int = Field(128, ge=1, description="Max tokens per line")
    require_finite_verb: bool = Field(True)
    max_pairs: int = Field(500_000, ge=0, description="Cap on training pairs (0=unlimited)")

    # Split
    dev_fraction: float = Field(0.1, ge=0.0, le=0.5)

    # Genre mapping
    genre_map: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_GENRE_MAP_EN))

    # Processing
    chunk_size: int = Field(1000, gt=0)

    @field_validator("input_path", "output_path", mode="before")
    @classmethod
    def resolve_paths(cls, v):
        return _resolve_path(v)

    @model_validator(mode="after")
    def _apply_language_defaults(self):
        return _apply_language_defaults(self)


class LLMAnnotationConfig(_BaseConfig):
    """Configuration for Steps 2-3: manual + LLM annotation."""

    corpus_path: Path = Field(..., description="Corpus directory with .train files")
    output_path: Path = Field(
        "data/pronoun_recovery/annotations/en",
        description="Base output dir (manual/, llm_raw/, llm_validated/ subdirs)",
    )
    seed_set_path: Optional[Path] = Field(
        None,
        description="Path to manual seed set JSONL (if already created)",
    )
    language: str = Field("en")

    # spaCy
    spacy_model: str = Field("en_core_web_trf")
    spacy_batch_size: int = Field(50, gt=0)

    # Sampling
    sample_size: int = Field(1000, gt=0, description="Candidate sequences to sample")
    genre_map: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_GENRE_MAP_EN))

    # LLM settings
    api_base_url: str = Field(
        "https://api.deepseek.com/v1",
        description="OpenAI-compatible API base URL",
    )
    api_key_env: str = Field("DEEPSEEK_API_KEY", description="Env var holding API key")
    model_name: str = Field("deepseek-chat", description="Model ID for completions")
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, gt=0)

    # Rate limiting
    requests_per_minute: int = Field(60, gt=0)
    tokens_per_minute: int = Field(100_000, gt=0)

    # Batch processing
    checkpoint_interval: int = Field(50, gt=0)
    max_retries: int = Field(3, ge=0)
    max_concurrent: int = Field(1, ge=1, description="Max concurrent LLM requests")

    @field_validator("corpus_path", "output_path", "seed_set_path", mode="before")
    @classmethod
    def resolve_paths(cls, v):
        return _resolve_path(v)

    @model_validator(mode="after")
    def _apply_language_defaults(self):
        return _apply_language_defaults(self)


class ValidationConfig(_BaseConfig):
    """Configuration for Step 4: annotation validation."""

    annotations_path: Path = Field(
        ..., description="Path to LLM annotation JSONL"
    )
    gold_path: Path = Field(
        ..., description="Path to manual gold-standard JSONL"
    )
    output_path: Path = Field(
        "data/pronoun_recovery/annotations/en/llm_validated",
        description="Validated annotation output dir",
    )
    language: str = Field("en")

    # Review sampling
    review_sample_size: int = Field(100, gt=0)
    random_fraction: float = Field(0.5, ge=0.0, le=1.0)
    high_insertion_fraction: float = Field(0.25, ge=0.0, le=1.0)
    flagged_fraction: float = Field(0.25, ge=0.0, le=1.0)

    # Gates
    min_precision: float = Field(0.9, ge=0.0, le=1.0)
    min_recall: float = Field(0.9, ge=0.0, le=1.0)
    min_feature_accuracy: float = Field(0.9, ge=0.0, le=1.0)

    @field_validator("annotations_path", "gold_path", "output_path", mode="before")
    @classmethod
    def resolve_paths(cls, v):
        return _resolve_path(v)


class ModelTrainingConfig(_BaseConfig):
    """Configuration for Step 5: DeBERTa sequence labeler training.

    Two training strategies, selected via ``training_source``:

    - ``"synthetic"``: Train on mechanically-generated pronoun-removal pairs.
      Suitable for English where pro-drop is rare and synthetic data provides
      clean labels at scale.  Uses ``synthetic_data_path``.

    - ``"llm_annotated"``: Train on LLM-annotated null-subject candidates.
      Suitable for Italian where pro-drop is the norm and the interesting
      signal is which pronoun belongs in each null-subject position.
      Uses ``annotation_data_path``.

    Both strategies fine-tune a pre-trained model (e.g. mDeBERTa) — neither
    trains from scratch.
    """

    training_source: str = Field(
        "synthetic",
        description=(
            "Training data strategy: 'synthetic' (English-style, train on "
            "mechanically removed pronouns) or 'llm_annotated' (Italian-style, "
            "train on LLM-annotated null subjects)"
        ),
    )
    synthetic_data_path: Optional[Path] = Field(
        None, description="Directory with train.jsonl / dev.jsonl (for training_source='synthetic')"
    )
    annotation_data_path: Optional[Path] = Field(
        None, description="Validated annotations JSONL (for training_source='llm_annotated')"
    )
    output_path: Path = Field(
        "data/pronoun_recovery/models/en",
        description="Model checkpoint output directory",
    )
    language: str = Field("en")

    # Model
    model_name: str = Field("microsoft/deberta-v3-base")
    num_labels: int = Field(7)

    # Synthetic training hyperparameters (training_source='synthetic')
    phase_a_lr: float = Field(3e-5)
    phase_a_batch_size: int = Field(32, gt=0)
    phase_a_epochs: int = Field(3, gt=0)
    phase_a_max_samples: int = Field(500_000, ge=0)
    phase_a_patience: int = Field(2, gt=0)

    # LLM-annotated training hyperparameters (training_source='llm_annotated')
    phase_b_lr: float = Field(1e-5)
    phase_b_batch_size: int = Field(16, gt=0)
    phase_b_epochs: int = Field(10, gt=0)
    phase_b_patience: int = Field(3, gt=0)

    # Class weighting
    weight_alpha: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Exponent for inverse-frequency class weights. "
            "1.0 = full inverse-frequency, 0.5 = sqrt, 0.0 = uniform."
        ),
    )
    focal_gamma: float = Field(
        0.0,
        ge=0.0,
        description=(
            "Focal loss focusing parameter. 0.0 = standard CE, "
            "2.0 = strong down-weighting of easy examples. "
            "Modulates loss by (1 - p_t)^gamma."
        ),
    )

    # Common
    fp16: bool = Field(True)
    warmup_ratio: float = Field(0.1, ge=0.0, le=1.0)
    max_seq_length: int = Field(256, gt=0)
    seed: int = Field(42)

    # Logging
    wandb_project: Optional[str] = Field("pronoun-recovery")
    wandb_run_name: Optional[str] = Field(None)

    # Gates
    min_detection_f1: float = Field(0.85, ge=0.0, le=1.0)
    min_label_accuracy: float = Field(0.80, ge=0.0, le=1.0)

    @field_validator("synthetic_data_path", "annotation_data_path", "output_path", mode="before")
    @classmethod
    def resolve_paths(cls, v):
        return _resolve_path(v)


class InsertionConfig(_BaseConfig):
    """Configuration for Stage 2: deterministic pronoun insertion."""

    input_path: Path = Field(..., description="Corpus directory with .train files")
    output_path: Path = Field(
        "data/pronoun_recovery/inserted/en",
        description="Output directory for pronoun-recovered .train files",
    )
    model_path: Path = Field(
        ..., description="Trained sequence labeler checkpoint directory"
    )
    language: str = Field("en")

    # spaCy
    spacy_model: str = Field("en_core_web_trf")
    spacy_batch_size: int = Field(50, gt=0)

    # Model inference
    model_batch_size: int = Field(64, gt=0)
    max_seq_length: int = Field(256, gt=0)
    device: Optional[str] = Field(None, description="torch device (auto-detected if None)")
    insertion_threshold: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Minimum P(any pronoun) to insert; 0.0 = always use argmax",
    )

    # Processing
    chunk_size: int = Field(1000, gt=0)
    genre_map: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_GENRE_MAP_EN))

    @field_validator("input_path", "output_path", "model_path", mode="before")
    @classmethod
    def resolve_paths(cls, v):
        return _resolve_path(v)

    @model_validator(mode="after")
    def _apply_language_defaults(self):
        return _apply_language_defaults(self)


class EuroparlAlignmentConfig(_BaseConfig):
    """Configuration for Europarl parallel alignment data generation.

    Uses awesome-align on EN-IT parallel Europarl sentences to extract
    high-quality pronoun labels by mapping English subject pronouns to
    Italian null-subject verbs.
    """

    europarl_en_path: Path = Field(
        ..., description="Path to Europarl English sentence file (one sentence per line)"
    )
    europarl_it_path: Path = Field(
        ..., description="Path to Europarl Italian sentence file (one sentence per line)"
    )
    output_path: Path = Field(
        "data/pronoun_recovery/europarl_aligned/it",
        description="Output directory for checkpoint JSONL",
    )
    language: str = Field("it")

    # Line range (for sharding / pilot)
    start_line: int = Field(0, ge=0, description="First line to process (0-based)")
    end_line: int = Field(50_000, ge=0, description="Last line (exclusive); 0=all")

    # spaCy models
    en_spacy_model: str = Field("en_core_web_trf")
    it_spacy_model: str = Field("it_core_news_lg")
    spacy_batch_size: int = Field(50, gt=0)

    # awesome-align
    align_model: str = Field(
        "aneuraz/awesome-align-with-co",
        description="HuggingFace model ID for awesome-align",
    )
    align_batch_size: int = Field(32, gt=0)

    # Pronoun extraction
    skip_all_it: bool = Field(
        True,
        description="Skip ALL occurrences of English 'it' (noisy: expletives, impersonals)",
    )

    # Quality filters
    min_tokens: int = Field(3, ge=1)
    max_tokens: int = Field(128, ge=1)
    max_length_ratio: float = Field(
        3.0,
        gt=0.0,
        description="Max ratio between EN and IT sentence lengths",
    )

    # Processing
    chunk_size: int = Field(1000, gt=0)
    checkpoint_interval: int = Field(5000, gt=0, description="Write checkpoint every N pairs")

    # Passage packing
    pack_passages: bool = Field(
        False,
        description="Group consecutive sentences into multi-sentence passages",
    )
    max_passage_words: int = Field(
        180,
        gt=0,
        description="Max words per passage (~234 subword tokens at 1.3x expansion)",
    )

    # Validation
    spot_check_n: int = Field(100, gt=0, description="Number of samples for spot-check review")

    # === Shared parse cache (DocBin) ===
    emit_target_docbin: bool = Field(
        False,
        description=(
            "If True, emit a spaCy DocBin of the target-language parses "
            "in 1:1 record order alongside aligned_checkpoint.jsonl. "
            "Downstream tree-detector feature extraction can then skip "
            "re-parsing clean_text. See TreeDetectorConfig.annotated_input_path."
        ),
    )

    @field_validator(
        "europarl_en_path", "europarl_it_path", "output_path", mode="before"
    )
    @classmethod
    def resolve_paths(cls, v):
        return _resolve_path(v)


class Seq2SeqTrainingConfig(_BaseConfig):
    """Configuration for seq2seq (mT5) pronoun recovery model training."""

    synthetic_data_path: Optional[Path] = Field(
        None, description="Directory with train.jsonl / dev.jsonl for Phase A (optional)"
    )
    annotation_data_path: Optional[Path] = Field(
        None, description="Validated annotations JSONL for Phase B"
    )
    output_path: Path = Field(
        "data/pronoun_recovery/models_seq2seq/en",
        description="Model checkpoint output directory",
    )
    language: str = Field("en")

    # Model
    model_name: str = Field("google/mt5-base")

    # Tokenization
    max_input_length: int = Field(256, gt=0)
    max_target_length: int = Field(256, gt=0)

    # Phase A: pretrain on synthetic data
    phase_a_lr: float = Field(3e-4)
    phase_a_batch_size: int = Field(16, gt=0)
    phase_a_epochs: int = Field(5, gt=0)
    phase_a_max_samples: int = Field(500_000, ge=0)
    phase_a_patience: int = Field(2, gt=0)

    # Phase B: finetune on annotations
    phase_b_lr: float = Field(1e-4)
    phase_b_batch_size: int = Field(8, gt=0)
    phase_b_epochs: int = Field(10, gt=0)
    phase_b_patience: int = Field(3, gt=0)

    # Generation
    num_beams: int = Field(4, gt=0)
    length_penalty: float = Field(1.0)

    # Common
    fp16: bool = Field(True)
    warmup_ratio: float = Field(0.1, ge=0.0, le=1.0)
    seed: int = Field(42)

    # Logging
    wandb_project: Optional[str] = Field("pronoun-recovery")
    wandb_run_name: Optional[str] = Field(None)

    @field_validator("synthetic_data_path", "annotation_data_path", "output_path", mode="before")
    @classmethod
    def resolve_paths(cls, v):
        return _resolve_path(v)


class Seq2SeqInsertionConfig(_BaseConfig):
    """Configuration for seq2seq-based pronoun insertion on full corpus."""

    input_path: Path = Field(..., description="Corpus directory with .train files")
    output_path: Path = Field(
        "data/pronoun_recovery/inserted_seq2seq/en",
        description="Output directory for pronoun-recovered .train files",
    )
    model_path: Path = Field(
        ..., description="Trained seq2seq model checkpoint directory"
    )
    language: str = Field("en")

    # Generation
    max_input_length: int = Field(256, gt=0)
    max_target_length: int = Field(256, gt=0)
    num_beams: int = Field(4, gt=0)
    length_penalty: float = Field(1.0)
    batch_size: int = Field(32, gt=0)

    # Processing
    chunk_size: int = Field(1000, gt=0)
    genre_map: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_GENRE_MAP_EN))

    @field_validator("input_path", "output_path", "model_path", mode="before")
    @classmethod
    def resolve_paths(cls, v):
        return _resolve_path(v)

    @model_validator(mode="after")
    def _apply_language_defaults(self):
        return _apply_language_defaults(self)


class TreeDetectorConfig(_BaseConfig):
    """Configuration for rule-based decision tree null subject detector.

    Two-stage architecture: (1) binary detection tree, (2) person/number
    from verb morphology.  Trains both a DecisionTreeClassifier and a
    HistGradientBoostingClassifier for comparison.
    """

    aligned_data_path: Path = Field(
        ...,
        description="Path to aligned_checkpoint.jsonl with gold markers",
    )
    output_path: Path = Field(
        "data/pronoun_recovery/tree_detector/it",
        description="Output directory for models, features, and reports",
    )
    language: str = Field("it")

    # spaCy
    it_spacy_model: str = Field("it_core_news_lg")
    spacy_batch_size: int = Field(50, gt=0)

    # === Shared parse cache ===
    annotated_input_path: Optional[Path] = Field(
        None,
        description=(
            "Optional path to a DocBin of target-language parses emitted "
            "by EuroparlAlignmentGenerator (with emit_target_docbin=True). "
            "When set, align_gold_data skips re-parsing clean_text and "
            "loads docs in 1:1 record order from "
            "{annotated_input_path}/target_parses.spacy."
        ),
    )

    # Classifier selection for final export
    classifier_type: str = Field(
        "decision_tree",
        description="Primary classifier: 'decision_tree' or 'gradient_boosted'",
    )

    # Decision tree hyperparameters
    max_depth: Optional[int] = Field(
        None,
        description="Max tree depth (None = unlimited, tuned via CV)",
    )
    min_samples_leaf: int = Field(
        5,
        ge=1,
        description="Minimum samples per leaf node",
    )

    # Gradient boosting hyperparameters
    n_estimators: int = Field(200, gt=0)
    learning_rate: float = Field(0.1, gt=0.0, le=1.0)
    gb_max_depth: Optional[int] = Field(
        None,
        description="Max tree depth for gradient boosting (None = unlimited)",
    )

    # Cross-validation
    cv_folds: int = Field(5, ge=2, le=20)
    test_fraction: float = Field(0.2, gt=0.0, lt=1.0)
    seed: int = Field(42)

    # Quality gates
    min_detection_f1: float = Field(0.80, ge=0.0, le=1.0)
    min_feature_accuracy: float = Field(0.95, ge=0.0, le=1.0)

    @field_validator(
        "aligned_data_path", "output_path", "annotated_input_path", mode="before",
    )
    @classmethod
    def resolve_paths(cls, v):
        return _resolve_path(v)


class TreeCrossEvalConfig(_BaseConfig):
    """Cross-domain evaluation of tree detectors across multiple data sources."""

    domain_paths: Dict[str, Path]  # name -> dir with features.parquet + labels.npy

    output_path: Path = Field(
        "data/pronoun_recovery/tree_detector/cross_eval",
        description="Output directory for cross-evaluation results",
    )

    # TTQ size sweep: list of training-set sizes (0 = use all available)
    ttq_sweep_sizes: List[int] = Field(
        default=[20000, 50000, 100000, 200000, 0],
    )

    # Standard training params
    test_fraction: float = Field(0.2, gt=0.0, lt=1.0)
    seed: int = Field(42)
    cv_folds: int = Field(5, ge=2, le=20)
    min_samples_leaf: int = Field(5, ge=1)
    n_estimators: int = Field(200, gt=0)
    learning_rate: float = Field(0.1, gt=0.0, le=1.0)
    gb_max_depth: Optional[int] = Field(None)
    max_depth: Optional[int] = Field(None)

    @field_validator("output_path", mode="before")
    @classmethod
    def resolve_output_path(cls, v):
        return _resolve_path(v)

    @field_validator("domain_paths", mode="before")
    @classmethod
    def resolve_domain_paths(cls, v):
        if isinstance(v, dict):
            return {k: _resolve_path(p) for k, p in v.items()}
        return v


def load_config(config_path: str, config_class: type) -> BaseModel:
    """Load a YAML config file into the specified Pydantic model."""
    import yaml

    path = Path(config_path)
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return config_class(**data)
