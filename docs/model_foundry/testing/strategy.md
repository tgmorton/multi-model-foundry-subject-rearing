# Model Foundry Testing Strategy

## Overview
Comprehensive testing strategy for the Model Foundry framework covering unit tests, integration tests, and end-to-end validation.

## Testing Hierarchy

```
model_foundry/
├── tests/
│   ├── unit/                    # Unit tests for individual components
│   │   ├── test_config.py       # Configuration validation
│   │   ├── test_data.py         # Data processing logic
│   │   ├── test_model.py        # Model creation
│   │   ├── test_utils.py        # Utility functions
│   │   ├── training/
│   │   │   ├── test_tokenization.py
│   │   │   ├── test_checkpointing.py
│   │   │   └── test_loop.py
│   ├── integration/             # Integration tests
│   │   ├── test_data_pipeline.py
│   │   ├── test_training_pipeline.py
│   │   └── test_checkpoint_recovery.py
│   ├── e2e/                     # End-to-end tests
│   │   └── test_full_training_run.py
│   ├── fixtures/                # Shared test fixtures
│   │   ├── configs/             # Sample config files
│   │   ├── datasets/            # Small test datasets
│   │   └── models/              # Tiny model checkpoints
│   └── conftest.py              # Pytest configuration
```

---

## Component-by-Component Testing Requirements

### 1. Configuration Module (`config.py`)

**Test Coverage:**
- ✅ Valid configuration parsing
- ✅ Invalid configuration rejection
- ✅ Field validation (types, ranges)
- ✅ Nested model validation
- ✅ Optional field defaults
- ✅ Edge cases (empty lists, None values)

**Critical Tests:**
```python
# Valid config loads successfully
# Invalid vocab_size (negative) raises ValidationError
# Missing required fields raises ValidationError
# Optional fields use defaults
# Nested configs validate correctly
# train_steps calculation logic
# warmup_steps calculation logic
```

**Mock Requirements:**
- None (pure validation logic)

---

### 2. Data Processing Module (`data.py`)

**Test Coverage:**
- ✅ Dataset validation and loading
- ✅ Streaming chunking logic
- ✅ Fixed-length chunk creation
- ✅ DataLoader creation
- ✅ Memory-mapped dataset loading
- ✅ Worker initialization for determinism
- ✅ Statistics calculation
- ✅ Edge cases (empty datasets, single example)

**Critical Tests:**
```python
# Validate tokenized dataset structure
# Chunk sequences with exact size
# Handle sequences shorter than chunk size
# Concatenate sequences to minimize waste
# Create DataLoader with correct batch size
# Worker init sets different seeds
# Calculate correct steps per epoch
# Handle missing test dataset gracefully
```

**Mock Requirements:**
- Mock datasets (HuggingFace Dataset objects)
- Mock tokenizers
- Temporary directories for saved data

**Test Data:**
- Small tokenized dataset (~100 sequences)
- Edge cases: 1 sequence, empty dataset
- Various sequence lengths

---

### 3. Model Module (`model.py`)

**Test Coverage:**
- ✅ Model creation with valid config
- ✅ Vocabulary size setting
- ✅ Architecture parameters
- ✅ Attention implementation switching
- ✅ Parameter counting
- ✅ Device placement

**Critical Tests:**
```python
# Create model with default config
# Model has correct vocab size
# Model has correct number of layers
# Model has correct hidden size
# Flash attention flag sets correctly
# Total parameter count matches expected
# Model can be moved to device
```

**Mock Requirements:**
- Mock config objects
- Small model configs for fast testing

---

### 4. Training - Tokenization (`training/tokenization.py`)

**Test Coverage:**
- ✅ Load HuggingFace tokenizer
- ✅ Load SentencePiece tokenizer
- ✅ Wrapper encode/decode functionality
- ✅ Special token handling
- ✅ Padding and truncation
- ✅ Batch tokenization
- ✅ Save/load roundtrip

**Critical Tests:**
```python
# Load standard tokenizer successfully
# Fall back to SentencePiece wrapper
# Encode text with special tokens
# Decode with/without special tokens
# Pad sequences to same length
# Handle attention masks correctly
# Save and reload tokenizer
# Batch processing maintains order
```

**Mock Requirements:**
- Mock SentencePiece processor
- Temporary tokenizer directories
- Sample tokenizer.model files

---

### 5. Training - Checkpointing (`training/checkpointing.py`)

**Test Coverage:**
- ✅ Checkpoint saving
- ✅ Checkpoint loading
- ✅ Metadata generation
- ✅ State preservation (optimizer, scheduler, RNG)
- ✅ Resume from latest checkpoint
- ✅ Schedule generation
- ✅ Checkpoint cleanup

**Critical Tests:**
```python
# Save checkpoint with all state
# Load checkpoint restores state
# Metadata includes all required fields
# RNG state preserved (reproducibility)
# Find latest checkpoint correctly
# Auto-generate checkpoint schedule
# Load checkpoint updates model weights
# Optimizer state restored correctly
# AMP scaler state handled
```

**Mock Requirements:**
- Mock models (small)
- Mock optimizers and schedulers
- Temporary checkpoint directories
- Mock config with schedule settings

**Critical Validation:**
- Reproducibility: Same RNG seed → same results after reload
- Completeness: All state saved and restored

---

### 6. Training - Loop (`training/loop.py`)

**Test Coverage:**
- ✅ Forward pass execution
- ✅ Backward pass and gradient computation
- ✅ Optimizer step
- ✅ Learning rate scheduling
- ✅ Gradient accumulation
- ✅ Gradient clipping
- ✅ AMP training path
- ✅ Memory monitoring
- ✅ Progress tracking
- ✅ OOM error handling
- ✅ Checkpoint saving integration

**Critical Tests:**
```python
# Single training step updates weights
# Gradient accumulation works correctly
# Gradient clipping applied when configured
# AMP path scales gradients
# Learning rate changes over time
# Loss decreases over steps (sanity check)
# OOM errors caught and handled
# Memory monitoring detects fragmentation
# Progress bar updates correctly
# Checkpoints saved at scheduled steps
```

**Mock Requirements:**
- Mock model (small, trainable)
- Mock dataloader (small batches)
- Mock checkpoint manager
- Mock data processor

**Performance Tests:**
- Memory usage stays within bounds
- Training throughput (steps/sec)

---

### 7. Main Trainer (`trainer.py`)

**Test Coverage:**
- ✅ Initialization with config
- ✅ Component orchestration
- ✅ Model initialization (Flash Attention fallback)
- ✅ Optimizer and scheduler setup
- ✅ Data preparation
- ✅ Training execution
- ✅ Checkpoint loading on resume
- ✅ Error handling and logging
- ✅ Environment snapshot

**Critical Tests:**
```python
# Initialize trainer with valid config
# Setup memory management on CUDA
# Calculate training parameters correctly
# Initialize all components
# Load tokenizer successfully
# Create dataloader
# Execute training loop
# Resume from checkpoint
# Handle training errors gracefully
# Save environment snapshot
```

**Mock Requirements:**
- Mock all subcomponents
- Mock CUDA availability
- Temporary directories

---

### 8. Utilities (`utils.py`)

**Test Coverage:**
- ✅ Find project root
- ✅ Git commit hash retrieval
- ✅ Seed setting
- ✅ Device detection

**Critical Tests:**
```python
# Find git root from nested path
# Get git commit hash
# Set seed makes results reproducible
# Detect CUDA correctly
# Fall back to CPU when no CUDA
```

**Mock Requirements:**
- Mock file systems
- Mock git commands
- Mock torch.cuda

---

### 9. Logging (`logging_utils.py`)

**Test Coverage:**
- ✅ Logger setup
- ✅ File handler creation
- ✅ Experiment-specific logging
- ✅ Multi-logger setup
- ✅ Log file listing

**Critical Tests:**
```python
# Create logger with correct name
# Log file created in correct location
# Multiple loggers don't interfere
# Log messages written correctly
# Timestamps formatted correctly
```

**Mock Requirements:**
- Temporary log directories

---

### 10. CLI (`cli.py`)

**Test Coverage:**
- ✅ Command parsing
- ✅ Config loading
- ✅ Each command executes
- ✅ Error handling for bad inputs
- ✅ Subprocess execution for preprocessing

**Critical Tests:**
```python
# Load valid config
# Reject invalid config
# Execute preprocess command
# Execute train command
# Execute validate command
# Handle missing files gracefully
```

**Mock Requirements:**
- Mock subprocess calls
- Mock trainer execution
- Temporary config files

---

## Integration Tests

### 1. Data Pipeline Integration
**Flow:** Raw data → Tokenization → Chunking → DataLoader

**Tests:**
```python
# Full data pipeline produces correct batches
# Chunking preserves token count
# DataLoader shuffles correctly
# Multiple workers don't cause issues
```

### 2. Training Pipeline Integration
**Flow:** Config → Model + Data → Training → Checkpoints

**Tests:**
```python
# Full training pipeline runs end-to-end
# Checkpoints saved at correct steps
# Resume from checkpoint continues correctly
# Loss logged to W&B (if configured)
```

### 3. Checkpoint Recovery Integration
**Flow:** Train → Save → Crash → Resume → Verify

**Tests:**
```python
# Save checkpoint mid-training
# Load checkpoint and resume
# Verify loss continues from same point
# Verify RNG state preserved (same next batch)
```

---

## End-to-End Tests

### Full Training Run (Tiny Model)
**Setup:**
- Tiny dataset (1000 sequences)
- Tiny model (2 layers, 64 hidden)
- 10 training steps
- 2 checkpoints

**Validation:**
- Training completes without errors
- Loss decreases
- Checkpoints created
- Model can generate text
- Memory usage reasonable

---

## Test Fixtures and Utilities

### Required Fixtures

```python
# conftest.py

@pytest.fixture
def tiny_config():
    """Minimal valid configuration for fast tests"""
    return ExperimentConfig(
        experiment_name="test_exp",
        data=DataConfig(
            source_corpus="test/data",
            training_corpus="test/data/train",
            batch_size=2,
            max_sequence_length=32
        ),
        tokenizer=TokenizerConfig(
            output_dir="test/tokenizer",
            vocab_size=1000
        ),
        model=ModelConfig(
            layers=2,
            embedding_size=64,
            hidden_size=64,
            intermediate_hidden_size=128,
            attention_heads=2,
            activation_function="gelu",
            dropout=0.1,
            attention_dropout=0.1
        ),
        training=TrainingConfig(
            output_dir="test/output",
            learning_rate=1e-4,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            epochs=1,
            train_steps=10,
            warmup_steps=2
        ),
        logging=LoggingConfig(
            level="INFO",
            dir="test/logs",
            use_wandb=False
        ),
        random_seed=42
    )

@pytest.fixture
def tiny_dataset():
    """Small tokenized dataset for testing"""
    from datasets import Dataset
    return Dataset.from_dict({
        'input_ids': [[1, 2, 3, 4, 5] * 10 for _ in range(100)]
    })

@pytest.fixture
def mock_tokenizer():
    """Simple mock tokenizer"""
    class MockTokenizer:
        vocab_size = 1000
        pad_token = "<pad>"
        eos_token = "</s>"
        pad_token_id = 0
        eos_token_id = 1

        def encode(self, text, add_special_tokens=True):
            return [1, 2, 3, 4, 5]

        def decode(self, ids, skip_special_tokens=True):
            return "test text"

        def save_pretrained(self, path):
            pass

    return MockTokenizer()

@pytest.fixture
def temp_workspace(tmp_path):
    """Temporary workspace with proper structure"""
    workspace = tmp_path / "workspace"
    (workspace / "data").mkdir(parents=True)
    (workspace / "models").mkdir(parents=True)
    (workspace / "logs").mkdir(parents=True)
    return workspace
```

---

## Test Execution Strategy

### 1. Local Development
```bash
# Fast unit tests only (< 30s)
pytest tests/unit/ -v

# Specific module
pytest tests/unit/test_data.py -v

# With coverage
pytest tests/unit/ --cov=model_foundry --cov-report=html
```

### 2. Pre-commit Checks
```bash
# Unit + integration tests (< 2min)
pytest tests/unit/ tests/integration/ -v
```

### 3. CI/CD Pipeline
```bash
# All tests including E2E (< 10min)
pytest tests/ -v --cov=model_foundry --cov-report=xml
```

### 4. GPU-Specific Tests
```bash
# Tests requiring CUDA
pytest tests/ -v -m gpu
```

---

## Coverage Goals

| Component | Target Coverage | Priority |
|-----------|----------------|----------|
| config.py | 95%+ | High |
| data.py | 90%+ | High |
| training/tokenization.py | 85%+ | High |
| training/checkpointing.py | 90%+ | Critical |
| training/loop.py | 85%+ | Critical |
| trainer.py | 80%+ | High |
| utils.py | 95%+ | Medium |
| logging_utils.py | 80%+ | Medium |
| cli.py | 70%+ | Medium |
| model.py | 85%+ | High |

**Overall Target: 85%+ coverage**

---

## Testing Tools

### Required Packages
```
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0
pytest-timeout>=2.1.0
hypothesis>=6.70.0  # Property-based testing
```

### Recommended Practices
1. **Fixtures over mocks** - Use real objects when possible
2. **Parameterized tests** - Test multiple scenarios efficiently
3. **Property-based testing** - Use Hypothesis for data processing
4. **Isolation** - Each test independent
5. **Fast feedback** - Unit tests < 0.1s each
6. **Determinism** - Set seeds, mock time-dependent code

---

## Common Test Patterns

### Testing PyTorch Models
```python
def test_model_forward_pass(tiny_config):
    model = create_model(tiny_config)
    batch = torch.randint(0, 1000, (2, 32))
    output = model(batch)
    assert output.logits.shape == (2, 32, 1000)
```

### Testing Data Processing
```python
@pytest.mark.parametrize("chunk_size,num_sequences", [
    (32, 100),
    (64, 50),
    (128, 25),
])
def test_chunking(chunk_size, num_sequences, tiny_dataset):
    processor = DataProcessor(config, base_dir)
    chunks = processor._create_chunked_dataset_streaming(
        tiny_dataset, chunk_size
    )
    assert all(len(chunk) == chunk_size for chunk in chunks)
```

### Testing Checkpointing
```python
def test_checkpoint_roundtrip(tiny_config, temp_workspace, tiny_model):
    manager = CheckpointManager(tiny_config, temp_workspace, "test_hash")

    # Save
    manager.save_checkpoint(tiny_model, tokenizer, optimizer, scheduler,
                          global_step=10, epoch=1)

    # Load
    loaded_model, _, step, epoch = manager.load_checkpoint(
        model_factory=lambda: create_model(tiny_config),
        device=torch.device("cpu"),
        optimizer=optimizer,
        lr_scheduler=scheduler
    )

    assert step == 10
    assert epoch == 1
    # Verify weights match
    for p1, p2 in zip(tiny_model.parameters(), loaded_model.parameters()):
        assert torch.allclose(p1, p2)
```

---

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/ --cov=model_foundry --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## Maintenance and Monitoring

### Regular Tasks
- 📊 **Weekly**: Review coverage reports
- 🔍 **Monthly**: Review flaky tests
- 🧹 **Quarterly**: Clean up deprecated tests
- 📈 **Release**: Full test suite + E2E validation

### Test Health Metrics
- ✅ Pass rate > 99%
- ⏱️ Unit test suite < 1 minute
- 📊 Coverage > 85%
- 🔄 Flaky test rate < 1%

---

## Next Steps

1. ✅ Create `tests/` directory structure
2. ✅ Implement `conftest.py` with fixtures
3. ✅ Write unit tests for each module (prioritize checkpointing)
4. ✅ Write integration tests
5. ✅ Setup CI/CD pipeline
6. ✅ Add coverage reporting
7. ✅ Document test running in README
