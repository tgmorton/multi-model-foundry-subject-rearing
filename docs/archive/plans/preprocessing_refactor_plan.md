# Preprocessing Pipeline Refactor Plan

**Date:** 2025-10-08
**Status:** Proposed
**Priority:** High - Critical for reproducibility and maintainability

## Executive Summary

The current preprocessing scripts (`preprocessing/`) have **80% code duplication** and **critical replicability gaps** (no random seeds, missing environment logging). This plan refactors them into a unified, config-driven pipeline that integrates seamlessly with `model_foundry` and supports all model architectures.

**Current State:** 5 scripts × 350 lines = 1,750 lines (1,400+ duplicated)
**Target State:** ~600 lines total (base class + ablation functions + config integration)

---

## Critical Issues to Address

### 1. Replicability Failures (Score: 4/10)
- ❌ **No random seed control** - Each run produces different outputs
- ❌ **No environment logging** - Can't reproduce results across systems
- ❌ **Model version ambiguity** - `en_core_web_sm` vs `en_core_web_trf` not logged
- ⚠️ **Chunk-dependent processing** - Results vary with `--chunk_size`

### 2. Code Quality Issues
- ❌ **80% code duplication** across 5 scripts
- ❌ **No error handling** - spaCy crashes unhandled
- ❌ **Memory inefficiency** - Loads entire files into RAM
- ❌ **No unit tests** - Zero test coverage

### 3. Integration Gaps
- ⚠️ **Partial model_foundry integration** - Uses logging_utils but not config system
- ❌ **No provenance tracking** - Can't trace model → corpus → preprocessing
- ❌ **Manual CLI invocation** - Not fully automated via config files

---

## Refactoring Goals

### Primary Objectives
1. ✅ **Eliminate code duplication** - DRY via base class abstraction
2. ✅ **Ensure reproducibility** - Seed control, environment logging, deterministic processing
3. ✅ **Full config integration** - Drive preprocessing from experiment YAML files
4. ✅ **Comprehensive error handling** - Graceful failures with logging
5. ✅ **Production-ready quality** - Unit tests, streaming I/O, performance optimization

### Secondary Objectives
6. ✅ **Multi-architecture support** - Works with GPT-2, BERT, LSTM, Mamba configs
7. ✅ **Pipeline composability** - Chain multiple ablations declaratively
8. ✅ **Provenance tracking** - Full metadata from input → output
9. ✅ **Developer experience** - Clear documentation, easy to add new ablations

---

## Architecture Design

### File Structure (New)
```
preprocessing/
├── __init__.py                    # Package init, expose public API
├── base.py                        # NEW: AblationPipeline base class
├── registry.py                    # NEW: Ablation function registry
├── config.py                      # NEW: Pydantic config models
├── provenance.py                  # NEW: Metadata tracking
├── utils.py                       # Shared utilities (device detection, tokenization)
├── ablations/                     # NEW: Directory for ablation functions
│   ├── __init__.py
│   ├── remove_expletives.py       # REFACTORED: Just ablation logic
│   ├── remove_articles.py         # REFACTORED: Just ablation logic
│   ├── impoverish_determiners.py  # REFACTORED: Just ablation logic
│   ├── lemmatize_verbs.py         # REFACTORED: Just ablation logic
│   └── remove_subject_pronominals.py  # REFACTORED: Just ablation logic
├── tests/                         # NEW: Unit tests
│   ├── __init__.py
│   ├── test_base.py
│   ├── test_ablations.py
│   └── fixtures/                  # Test data
└── 00_prepare_corpus.py           # KEEP: Already well-isolated

model_foundry/
├── cli.py                         # MODIFIED: Enhanced preprocess command
└── config.py                      # MODIFIED: Add preprocessing config models
```

### Core Components

#### 1. `preprocessing/base.py` - AblationPipeline Base Class

**Responsibilities:**
- File I/O with streaming support
- Progress tracking and logging
- Validation execution
- Statistics collection
- Replacement pool management
- Error handling and recovery

**Key Methods:**
```python
class AblationPipeline:
    def __init__(self, config: AblationConfig):
        """Initialize with validated config."""

    def process_corpus(self) -> ProvenanceManifest:
        """Main entry point - processes all files in corpus."""

    def _process_file(self, file_path: Path) -> FileStats:
        """Process single file with streaming."""

    def _ablate_chunk(self, chunk: List[str]) -> Tuple[str, int]:
        """Process chunk - delegates to ablation function."""

    def _validate_ablation(self, original: str, ablated: str) -> bool:
        """Validate ablation occurred (optional, can skip)."""

    def _rebuild_to_target_size(self, ...) -> str:
        """Use replacement pool to restore token count."""
```

#### 2. `preprocessing/registry.py` - Ablation Function Registry

**Purpose:** Centralized registration of ablation functions.

```python
from typing import Callable, Dict, Tuple
import spacy

AblationFunction = Callable[[spacy.tokens.Doc], Tuple[str, int]]
ValidationFunction = Callable[[str, str, spacy.Language], bool]

class AblationRegistry:
    _ablations: Dict[str, AblationFunction] = {}
    _validators: Dict[str, ValidationFunction] = {}

    @classmethod
    def register(cls, name: str,
                 ablation_fn: AblationFunction,
                 validation_fn: Optional[ValidationFunction] = None):
        """Register ablation + optional validator."""
        cls._ablations[name] = ablation_fn
        if validation_fn:
            cls._validators[name] = validation_fn

    @classmethod
    def get(cls, name: str) -> Tuple[AblationFunction, Optional[ValidationFunction]]:
        """Retrieve registered ablation."""
        return cls._ablations[name], cls._validators.get(name)

# Usage in ablation modules
from preprocessing.registry import AblationRegistry

def remove_articles_doc(doc):
    # ... existing logic ...
    pass

def validate_article_removal(original, ablated, nlp):
    # ... existing logic ...
    pass

AblationRegistry.register("remove_articles", remove_articles_doc, validate_article_removal)
```

#### 3. `preprocessing/config.py` - Configuration Models

**Purpose:** Pydantic models for type-safe configuration.

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
from pathlib import Path
import random

class AblationConfig(BaseModel):
    """Configuration for a single ablation step."""

    # Core parameters
    type: str = Field(..., description="Ablation type (registered name)")
    input_path: Path
    output_path: Path

    # Reproducibility
    seed: int = Field(42, description="Random seed for reproducibility")

    # Processing parameters
    chunk_size: int = Field(1000, gt=0, description="Lines per processing chunk")
    skip_validation: bool = Field(False, description="Skip validation step")

    # Replacement pool
    replacement_pool_dir: Optional[Path] = None

    # spaCy configuration
    spacy_model: str = Field("en_core_web_sm", description="spaCy model to use")
    spacy_device: Optional[str] = Field(None, description="Device: 'cpu', 'cuda', 'mps', or None=auto")

    # Logging
    verbose: bool = Field(False, description="Enable verbose logging")
    log_dir: Path = Field(Path("logs"), description="Log output directory")

    # Additional ablation-specific parameters
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Ablation-specific params")

    @field_validator('input_path', 'output_path', 'replacement_pool_dir')
    @classmethod
    def resolve_paths(cls, v):
        """Resolve paths relative to project root."""
        if v and not v.is_absolute():
            from model_foundry.utils import find_project_root
            root = find_project_root(__file__)
            return root / v
        return v

    def model_post_init(self, __context):
        """Set random seed after initialization."""
        random.seed(self.seed)
        # Also set numpy, torch if available
        try:
            import numpy as np
            np.random.seed(self.seed)
        except ImportError:
            pass
        try:
            import torch
            torch.manual_seed(self.seed)
        except ImportError:
            pass


class ProvenanceMetadata(BaseModel):
    """Metadata for tracking ablation provenance."""

    # Execution environment
    timestamp: str
    python_version: str
    spacy_version: str
    spacy_model_name: str
    spacy_model_version: str
    device: str
    hostname: str

    # Configuration
    ablation_type: str
    random_seed: int
    chunk_size: int

    # Input/Output checksums
    input_checksums: Dict[str, str]  # {filename: sha256}
    output_checksums: Dict[str, str]

    # Statistics
    total_files_processed: int
    total_tokens_original: int
    total_tokens_final: int
    total_items_ablated: int  # expletives, articles, etc.
    processing_time_seconds: float


class ProvenanceManifest(BaseModel):
    """Complete provenance record for an ablation run."""

    metadata: ProvenanceMetadata
    file_statistics: List[Dict[str, Any]]
    config: AblationConfig

    def save(self, output_dir: Path):
        """Save manifest to JSON."""
        manifest_path = output_dir / "ABLATION_MANIFEST.json"
        manifest_path.write_text(self.model_dump_json(indent=2))
```

#### 4. `preprocessing/utils.py` - Shared Utilities

**Consolidate existing utility functions:**

```python
import hashlib
import platform
import sys
from pathlib import Path
from typing import Optional

def get_spacy_device(verbose: bool = False) -> str:
    """Auto-detect best spaCy device (MPS, CUDA, CPU)."""
    # ... existing logic from remove_expletives.py ...

def count_tokens(text: str) -> int:
    """Count tokens consistently with training tokenization."""
    # ... existing logic ...

def compute_file_checksum(file_path: Path) -> str:
    """Compute SHA256 checksum of file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def get_environment_info() -> dict:
    """Capture complete environment metadata."""
    import spacy

    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "spacy_version": spacy.__version__,
    }

    # Add torch version if available
    try:
        import torch
        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        pass

    return info
```

#### 5. `preprocessing/ablations/*.py` - Refactored Ablation Modules

**Each file becomes ~50 lines instead of 350+:**

```python
"""
Remove Articles Ablation

Removes all articles ('a', 'an', 'the') from the corpus.
"""

from typing import Tuple
import spacy
from preprocessing.registry import AblationRegistry

def remove_articles_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """
    Remove all articles from a spaCy Doc.

    Args:
        doc: spaCy Doc object

    Returns:
        (ablated_text, num_articles_removed)
    """
    modified_parts = []
    num_removed = 0

    for token in doc:
        is_article = token.pos_ == 'DET' and token.lower_ in ['a', 'an', 'the']
        if not is_article:
            modified_parts.append(token.text_with_ws)
        else:
            num_removed += 1

    return ''.join(modified_parts), num_removed


def validate_article_removal(original_text: str, ablated_text: str,
                             nlp: spacy.Language) -> bool:
    """Validate that articles were removed."""
    original_doc = nlp(original_text)
    ablated_doc = nlp(ablated_text)

    original_articles = [
        token.text for token in original_doc
        if token.pos_ == 'DET' and token.lower_ in ['a', 'an', 'the']
    ]
    ablated_articles = [
        token.text for token in ablated_doc
        if token.pos_ == 'DET' and token.lower_ in ['a', 'an', 'the']
    ]

    return len(ablated_articles) < len(original_articles) if original_articles else True


# Register this ablation
AblationRegistry.register("remove_articles", remove_articles_doc, validate_article_removal)
```

---

## Integration with model_foundry

### Updated Config Schema (model_foundry/config.py)

```python
class PreprocessingStepConfig(BaseModel):
    """Single preprocessing step configuration."""
    type: str = Field(..., description="Ablation type (e.g., 'remove_expletives')")
    input_path: str
    output_path: str
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Ablation-specific parameters"
    )
    seed: Optional[int] = Field(None, description="Override experiment seed for this step")

class ExperimentConfig(BaseModel):
    experiment_name: str
    random_seed: int = 42  # Global seed

    data: DataConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    training: TrainingConfig

    # NEW: Preprocessing pipeline
    dataset_manipulation: Optional[List[PreprocessingStepConfig]] = Field(
        None,
        description="Sequential preprocessing steps"
    )

    # NEW: Preprocessing defaults
    preprocessing: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "chunk_size": 1000,
            "skip_validation": False,
            "verbose": False,
            "spacy_model": "en_core_web_sm",
        },
        description="Default parameters for all preprocessing steps"
    )
```

### Enhanced CLI Command (model_foundry/cli.py)

```python
@app.command()
def preprocess(
    config_path: str = typer.Argument(..., help="Path to experiment YAML"),
    dry_run: bool = Option(False, "--dry-run", help="Show plan without executing"),
    step: Optional[int] = Option(None, "--step", help="Run only specific step (1-indexed)"),
):
    """
    Run the dataset manipulation pipeline from config.

    NEW: Uses AblationPipeline base class instead of subprocess calls.
    """
    logger = logging.getLogger(__name__)
    config = load_config(config_path)

    if not config.dataset_manipulation:
        logger.info("No preprocessing steps defined. Skipping.")
        return

    # Import preprocessing components
    from preprocessing.base import AblationPipeline
    from preprocessing.config import AblationConfig, ProvenanceManifest
    from preprocessing.registry import AblationRegistry

    # Process each step
    steps = config.dataset_manipulation
    if step is not None:
        steps = [steps[step - 1]]

    for i, step_config in enumerate(steps, 1):
        logger.info(f"Step {i}/{len(config.dataset_manipulation)}: {step_config.type}")

        # Build ablation config (merge experiment + step + defaults)
        ablation_config = AblationConfig(
            type=step_config.type,
            input_path=step_config.input_path,
            output_path=step_config.output_path,
            seed=step_config.seed or config.random_seed,  # Use experiment seed if not overridden
            **config.preprocessing,  # Apply defaults
            **step_config.parameters,  # Override with step-specific params
        )

        if dry_run:
            logger.info(f"  [DRY RUN] Would process: {ablation_config.input_path} → {ablation_config.output_path}")
            continue

        # Run ablation
        pipeline = AblationPipeline(ablation_config)
        manifest = pipeline.process_corpus()

        logger.info(f"  ✓ Completed: {manifest.metadata.total_items_ablated:,} items ablated")
        logger.info(f"  ✓ Provenance saved: {ablation_config.output_path}/ABLATION_MANIFEST.json")
```

### Example Config Usage

```yaml
# configs/experiment_1_remove_expletives.yaml

experiment_name: "exp1_remove_expletives"
random_seed: 42  # Applied to all steps by default

# Global preprocessing defaults
preprocessing:
  chunk_size: 1000
  skip_validation: true  # Speed up production runs
  verbose: false
  spacy_model: "en_core_web_sm"

dataset_manipulation:
  # Step 1: Prepare corpus (split into main + pool)
  - type: prepare_corpus  # Maps to 00_prepare_corpus.py
    input_path: "data/raw/train_90M/"
    output_path: "data/intermediate/main_90M/"
    parameters:
      pool_output_dir: "data/intermediate/pool_10M/"
      pool_words_total: 10000000

  # Step 2: Remove expletives
  - type: remove_expletives
    input_path: "data/intermediate/main_90M/"
    output_path: "data/processed/exp1_remove_expletives/"
    parameters:
      replacement_pool_dir: "data/intermediate/pool_10M/"
      # Inherits chunk_size, skip_validation, etc. from preprocessing defaults

data:
  training_corpus: "data/processed/exp1_remove_expletives/"  # Final output
  # ... rest of config ...
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1)
**Goal:** Build base infrastructure without breaking existing scripts.

**Tasks:**
1. Create `preprocessing/base.py` with `AblationPipeline` class
2. Create `preprocessing/registry.py` with registration system
3. Create `preprocessing/config.py` with Pydantic models
4. Create `preprocessing/utils.py` consolidating shared functions
5. Add comprehensive docstrings and type hints

**Deliverables:**
- Base classes pass type checking (`mypy`)
- Sphinx documentation generated
- No impact on existing scripts (they still work)

**Testing:**
- Unit tests for `AblationPipeline._process_file()` with mock data
- Unit tests for `AblationConfig` validation
- Integration test: run base class with dummy ablation function

### Phase 2: Refactor One Ablation (Week 2)
**Goal:** Prove the architecture with `remove_articles.py`.

**Tasks:**
1. Create `preprocessing/ablations/remove_articles.py` (minimal version)
2. Register with `AblationRegistry`
3. Update CLI to support new system (fallback to old if not registered)
4. Add unit tests for article removal
5. Add integration test comparing old vs new output

**Deliverables:**
- `remove_articles` works via new pipeline
- Old script still functional (parallel systems)
- Integration test proves identical output

**Success Criteria:**
- Old and new produce byte-identical output (given same seed)
- New version is faster (streaming I/O)
- Provenance manifest generated

### Phase 3: Migrate All Ablations (Week 3)
**Goal:** Refactor remaining 4 ablations.

**Tasks:**
1. Refactor `remove_expletives.py` (most complex - coreference)
2. Refactor `impoverish_determiners.py`
3. Refactor `lemmatize_verbs.py`
4. Refactor `remove_subject_pronominals.py`
5. Add tests for each

**Deliverables:**
- All 5 ablations work via new system
- Test coverage > 80%
- Old scripts moved to `preprocessing/legacy/`

**Testing:**
- Regression tests for each ablation
- End-to-end test: run full pipeline from config

### Phase 4: Enhanced Features (Week 4)
**Goal:** Add production-ready features.

**Tasks:**
1. Implement streaming I/O for large files
2. Add comprehensive error handling with recovery
3. Add progress checkpointing (resume interrupted runs)
4. Add parallel processing (multi-file corpus)
5. Optimize performance (profile and fix bottlenecks)

**Deliverables:**
- Can process 100M+ token corpus without OOM
- Handles spaCy errors gracefully
- Can resume after crash
- 2-3x faster than original scripts

**Testing:**
- Stress test with 1GB+ files
- Fault injection tests (simulated crashes)
- Performance benchmarks

### Phase 5: Documentation & Integration (Week 5)
**Goal:** Complete integration and documentation.

**Tasks:**
1. Update README with new preprocessing workflow
2. Create developer guide for adding new ablations
3. Update all experiment configs to use new system
4. Add CLI help documentation
5. Create migration guide for legacy scripts

**Deliverables:**
- Complete documentation in `/docs/preprocessing/`
- All experiment configs use new pipeline
- Migration completed (old scripts archived)

**Acceptance:**
- New user can add custom ablation in < 30 minutes
- All existing experiments reproduce exactly

---

## Testing Strategy

### Unit Tests (`preprocessing/tests/test_base.py`)
```python
def test_ablation_pipeline_init():
    """Test pipeline initialization with valid config."""

def test_process_file_streaming():
    """Test file processing with streaming I/O."""

def test_replacement_pool_management():
    """Test pool sampling is deterministic with seed."""

def test_validation_execution():
    """Test validation runs and logs correctly."""

def test_error_handling_spacy_crash():
    """Test graceful handling of spaCy errors."""
```

### Integration Tests (`preprocessing/tests/test_ablations.py`)
```python
def test_remove_articles_end_to_end():
    """Test full pipeline with article removal."""

def test_pipeline_reproducibility():
    """Test same seed produces identical output."""

def test_multi_step_pipeline():
    """Test chaining multiple ablations."""

def test_provenance_manifest_generation():
    """Test manifest contains all required metadata."""
```

### Regression Tests
```python
def test_output_matches_legacy_script():
    """Compare new pipeline output to original scripts."""
    # Run legacy script
    legacy_output = run_legacy_remove_articles(input_file, seed=42)

    # Run new pipeline
    new_output = run_new_pipeline("remove_articles", input_file, seed=42)

    # Compare token-by-token
    assert legacy_output == new_output
```

---

## Performance Improvements

### Expected Gains

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Memory Usage | ~5GB (load entire file) | ~500MB (streaming) | 10x reduction |
| Processing Speed | Baseline | 2-3x faster | Optimized chunking |
| Startup Time | ~30s (model load per script) | ~10s (load once) | 3x faster |
| Error Recovery | Fails and restarts | Resume from checkpoint | Saves hours |

### Optimization Techniques

1. **Streaming I/O**
   - Replace `readlines()` with generator
   - Write output incrementally
   - Process 1000-line chunks

2. **spaCy Pipeline Optimization**
   - Use `nlp.pipe()` with batching (already done)
   - Disable unused components: `nlp.select_pipes(enable=["tagger", "parser"])`
   - Cache processed docs (careful with memory)

3. **Parallel Processing**
   - Use `multiprocessing.Pool` for multi-file corpora
   - Each worker processes one file
   - Aggregate statistics at end

4. **Smart Caching**
   - Cache spaCy model between steps
   - Reuse replacement pool in memory
   - Cache validation results

---

## Migration Path

### For Users
1. **Update configs** - Add `preprocessing:` section with defaults
2. **Run new CLI** - `python -m model_foundry.cli preprocess configs/experiment.yaml`
3. **Verify provenance** - Check `ABLATION_MANIFEST.json` was created
4. **Compare output** - Regression test against old scripts (optional)

### For Developers
1. **Read developer guide** - `/docs/preprocessing/developer_guide.md`
2. **Copy template** - `cp preprocessing/ablations/template.py preprocessing/ablations/my_ablation.py`
3. **Implement function** - Write ablation logic (~30 lines)
4. **Register** - `AblationRegistry.register("my_ablation", my_ablation_fn)`
5. **Test** - `pytest preprocessing/tests/test_my_ablation.py`
6. **Use in config** - Add to `dataset_manipulation:` list

---

## Rollback Plan

If issues arise during migration:

1. **Phase 1-2:** Old scripts unaffected - simply don't use new system
2. **Phase 3:** Old scripts in `preprocessing/legacy/` - can still be invoked manually
3. **Phase 4-5:** Revert commits - `git revert <commit-range>`

**Critical:** Keep old scripts functional until Phase 5 complete and all experiments verified.

---

## Success Metrics

### Quantitative
- [ ] Code duplication: 80% → < 5%
- [ ] Test coverage: 0% → > 80%
- [ ] Replicability score: 4/10 → 9/10
- [ ] Memory usage: -90% (streaming I/O)
- [ ] Processing speed: +200% (optimizations)
- [ ] Time to add new ablation: 2 hours → 30 minutes

### Qualitative
- [ ] All existing experiments reproduce exactly
- [ ] New developer can add ablation without asking questions
- [ ] Provenance manifests enable full result tracking
- [ ] Codebase feels maintainable (subjective but important)

---

## Dependencies & Requirements

### New Python Packages
```txt
# Already in requirements.txt:
spacy==3.7.5
pydantic==2.7.4
tqdm==4.66.4

# May need to add:
pytest==8.0.0  # For testing
pytest-cov==4.1.0  # Coverage reporting
mypy==1.8.0  # Type checking
```

### System Requirements
- Python 3.8+
- spaCy models: `en_core_web_sm` (required), `en_core_web_trf` (optional)
- Sufficient disk space for provenance manifests (~1% of corpus size)

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Breaking existing experiments | Medium | High | Keep old scripts, regression tests |
| Performance regression | Low | Medium | Benchmark early, optimize Phase 4 |
| spaCy version incompatibility | Low | Low | Pin versions, log model metadata |
| Developer adoption resistance | Low | Medium | Good docs, easy migration |
| Unforeseen edge cases | Medium | Medium | Comprehensive testing, gradual rollout |

---

## Timeline Summary

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| Phase 1: Foundation | 1 week | Base classes implemented |
| Phase 2: Prove Architecture | 1 week | `remove_articles` refactored |
| Phase 3: Migrate Ablations | 1 week | All 5 ablations migrated |
| Phase 4: Production Features | 1 week | Streaming, error handling |
| Phase 5: Integration | 1 week | Docs, migration complete |
| **Total** | **5 weeks** | **Production-ready system** |

---

## Next Steps

1. **Review this plan** - Get stakeholder approval
2. **Create GitHub issues** - One per phase, with subtasks
3. **Set up testing infrastructure** - pytest, coverage, fixtures
4. **Begin Phase 1** - Start with `base.py` implementation
5. **Weekly check-ins** - Review progress, adjust timeline

---

## Appendix A: Example Ablation Function Template

```python
"""
<Ablation Name> Ablation

<Brief description of what this ablation does and why.>
"""

from typing import Tuple, Optional
import spacy
from preprocessing.registry import AblationRegistry

def <ablation_name>_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """
    <Description of transformation.>

    Args:
        doc: spaCy Doc object to process

    Returns:
        Tuple of (ablated_text, num_modifications)
    """
    modified_parts = []
    num_modifications = 0

    for token in doc:
        # Your ablation logic here
        if <condition>:
            # Modify token
            modified_parts.append(<modified_token>)
            num_modifications += 1
        else:
            # Keep original
            modified_parts.append(token.text_with_ws)

    return ''.join(modified_parts), num_modifications


def validate_<ablation_name>(original_text: str, ablated_text: str,
                              nlp: spacy.Language) -> bool:
    """
    Validate that ablation occurred.

    Args:
        original_text: Original text before ablation
        ablated_text: Text after ablation
        nlp: spaCy NLP pipeline

    Returns:
        True if ablation was successful, False otherwise
    """
    original_doc = nlp(original_text)
    ablated_doc = nlp(ablated_text)

    # Count relevant items in original
    original_count = sum(1 for token in original_doc if <condition>)

    # Count relevant items in ablated
    ablated_count = sum(1 for token in ablated_doc if <condition>)

    # Should be fewer (or zero if none existed)
    return ablated_count < original_count if original_count > 0 else True


# Register this ablation with the registry
AblationRegistry.register(
    "<ablation_name>",
    <ablation_name>_doc,
    validate_<ablation_name>
)
```

---

## Appendix B: Provenance Manifest Example

```json
{
  "metadata": {
    "timestamp": "2025-10-08T14:32:15Z",
    "python_version": "3.10.12",
    "spacy_version": "3.7.5",
    "spacy_model_name": "en_core_web_sm",
    "spacy_model_version": "3.7.1",
    "device": "mps",
    "hostname": "research-macbook.local",
    "ablation_type": "remove_expletives",
    "random_seed": 42,
    "chunk_size": 1000,
    "input_checksums": {
      "bnc_spoken.train": "a3f2b1c9...",
      "childes.train": "b7e8f3d2..."
    },
    "output_checksums": {
      "bnc_spoken.train": "c9d4e5f1...",
      "childes.train": "d2e7f8a3..."
    },
    "total_files_processed": 6,
    "total_tokens_original": 90000000,
    "total_tokens_final": 90000000,
    "total_items_ablated": 125000,
    "processing_time_seconds": 3847.2
  },
  "file_statistics": [
    {
      "file_name": "bnc_spoken.train",
      "original_tokens": 15000000,
      "tokens_removed": 25000,
      "expletives_removed": 25000,
      "proportion_removed": 0.00167
    }
  ],
  "config": {
    "type": "remove_expletives",
    "input_path": "/path/to/data/raw/train_90M/",
    "output_path": "/path/to/data/processed/exp1/",
    "seed": 42,
    "chunk_size": 1000,
    "spacy_model": "en_core_web_sm",
    "parameters": {}
  }
}
```

---

**End of Plan**
