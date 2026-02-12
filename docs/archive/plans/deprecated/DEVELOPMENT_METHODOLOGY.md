# Multi-Architecture Expansion: Development Methodology

**Purpose:** Establish consistent development practices for implementing multi-architecture support in the Model Foundry framework across all development sessions.

**Date:** 2025-10-02

---

## Project Context

This document defines the implementation methodology for expanding Model Foundry to support RNN, LSTM, and BERT architectures alongside the existing GPT-2 implementation, as specified in [MULTI_ARCHITECTURE_EXPANSION.md](./MULTI_ARCHITECTURE_EXPANSION.md).

---

## Core Principles

### 1. Incremental Implementation with Testing
- Implement features in small, testable units
- Write or update tests **concurrently** with implementation, not after
- Commit to repository after each completed, tested component
- Never batch multiple phases into a single commit

### 2. Phase-Based Development
Each phase consists of:
1. **Implementation** - Code the feature
2. **Testing** - Write/update tests to validate
3. **Documentation** - Update relevant docs
4. **Commit** - Commit all three together

**Phase Order:**
- Phase 1: Architecture Abstraction (GPT-2 refactor)
- Phase 2: BERT Implementation + WordPiece Tokenizer
- Phase 3: LSTM Implementation + Appropriate Tokenizer
- Phase 4: Additional RNN variants (GRU, vanilla RNN)
- Phase 5: Full integration and cross-architecture validation

### 3. Backwards Compatibility Policy
- **Breaking changes are acceptable** for this implementation
- Existing config specifications must remain expressible in new system
- Migration of existing configs will be handled separately
- **All new configs MUST explicitly specify `architecture` field**
- Fail with clear error if `architecture` field is missing

### 4. Testing Requirements
- Follow testing strategy in [docs/model_foundry/testing/strategy.md](../docs/model_foundry/testing/strategy.md)
- **Test-Driven Development:** Write tests before or alongside implementation
- Minimum coverage thresholds (from strategy doc):
  - Critical components (checkpointing, training loop): 85-90%+
  - Core components (config, data, model): 85-90%+
  - Supporting components (utils, logging): 70-80%+

### 5. Documentation Updates
- Update existing documentation at **end of each phase**
- Do not create new documentation files unless explicitly necessary
- Keep documentation synchronized with implementation
- Document breaking changes clearly

---

## Implementation Phases

### Phase 1: Architecture Abstraction Layer

**Objective:** Create abstraction layer and refactor GPT-2 without breaking existing functionality.

**Components to Implement:**
1. `model_foundry/architectures/` module structure
2. `base.py` - Abstract base classes for language models
3. `__init__.py` - Registry pattern and factory function
4. `gpt.py` - Migrate existing GPT-2 logic
5. `utils.py` - Shared utilities for architectures
6. Update `model_foundry/model.py` to use factory pattern
7. Update `model_foundry/config.py` to support `architecture` field

**Testing Requirements:**
- Unit tests for base classes
- Unit tests for model registry
- Unit tests for GPT-2 architecture wrapper
- Integration test: existing GPT-2 functionality still works
- Validation test: config without `architecture` fails with clear error

**Documentation Updates:**
- Update `docs/model_foundry/architecture/` if exists
- Update any model creation documentation

**Commit Message Format:**
```
Phase 1: Implement architecture abstraction layer

- Create base classes for multi-architecture support
- Implement model registry pattern
- Refactor GPT-2 into architecture module
- Update config schema to require architecture field
- Add tests for architecture abstraction
```

**Tokenizer for Testing:**
- Use existing SentencePiece tokenizer
- Can test with existing tokenizer infrastructure

---

### Phase 2: BERT Implementation

**Objective:** Add BERT architecture with masked language modeling support.

**Components to Implement:**
1. `model_foundry/architectures/bert.py` - BERT model wrapper
2. `model_foundry/data_collators.py` - Causal LM and Masked LM collators
3. Update `model_foundry/config.py` - Add BERT-specific config, MLM training objective
4. Update `model_foundry/data.py` - Use appropriate data collator
5. `model_foundry/tokenizer/tokenizer_factory.py` - Multi-tokenizer support
6. Implement WordPiece tokenizer training in factory
7. Update `model_foundry/tokenizer/train_tokenizer.py` - Use factory

**Testing Requirements:**
- Unit tests for BERT architecture
- Unit tests for MaskedLMDataCollator
- Unit tests for WordPiece tokenizer training
- Unit tests for tokenizer factory
- Integration test: Train small BERT model with MLM objective
- Validation test: Loss decreases over training steps
- E2E test: Full BERT training pipeline with WordPiece tokenizer

**Documentation Updates:**
- Update tokenizer documentation
- Update configuration schema documentation
- Update training objective documentation

**Commit Message Format:**
```
Phase 2: Implement BERT architecture and WordPiece tokenizer

- Add BERT model with masked language modeling
- Implement data collators for causal and masked LM
- Add WordPiece tokenizer training
- Implement tokenizer factory pattern
- Update config schema for BERT and MLM
- Add comprehensive tests for BERT pipeline
```

**Example Config to Create:**
- `configs/test_bert_tiny.yaml` - Minimal BERT config for testing

---

### Phase 3: LSTM Implementation

**Objective:** Add LSTM architecture with both causal and masked LM support.

**Components to Implement:**
1. `model_foundry/architectures/rnn.py` - LSTM model implementation
2. Update `model_foundry/config.py` - Add RNN-specific config
3. Add BPE tokenizer training to factory (if not already present)
4. Implement bidirectional and unidirectional LSTM variants

**Testing Requirements:**
- Unit tests for LSTM forward pass
- Unit tests for unidirectional LSTM (causal LM)
- Unit tests for bidirectional LSTM (masked LM)
- Integration test: Train small LSTM with causal LM
- Integration test: Train small LSTM with masked LM
- Validation test: Gradient flow check
- Validation test: Loss convergence

**Documentation Updates:**
- Update architecture documentation
- Update RNN-specific training recommendations

**Commit Message Format:**
```
Phase 3: Implement LSTM architecture

- Add LSTM model with uni/bidirectional support
- Support both causal and masked LM objectives
- Add RNN-specific configuration
- Add comprehensive LSTM tests
- Update architecture documentation
```

**Example Configs to Create:**
- `configs/test_lstm_causal_tiny.yaml`
- `configs/test_lstm_bidirectional_tiny.yaml`

---

### Phase 4: Additional RNN Variants

**Objective:** Add GRU and vanilla RNN support.

**Components to Implement:**
1. Extend `model_foundry/architectures/rnn.py` - Add GRU and RNN variants
2. Update config to support `rnn_type` parameter
3. Character-level tokenizer (if needed for specific experiments)

**Testing Requirements:**
- Unit tests for GRU
- Unit tests for vanilla RNN
- Integration tests for each variant
- Comparison test: Verify architectural differences

**Documentation Updates:**
- Update RNN architecture documentation

**Commit Message Format:**
```
Phase 4: Add GRU and vanilla RNN variants

- Implement GRU architecture
- Implement vanilla RNN architecture
- Add rnn_type configuration parameter
- Add tests for RNN variants
```

---

### Phase 5: Integration and Validation

**Objective:** Ensure all architectures work together and validate cross-architecture compatibility.

**Components to Implement:**
1. Cross-architecture validation suite
2. Performance benchmarking utilities
3. Final documentation updates
4. Update evaluation framework for bidirectional models

**Testing Requirements:**
- E2E test: Train all architectures on same small dataset
- Validation test: Compare convergence characteristics
- Performance test: Memory usage profiling
- Performance test: Training speed benchmarking

**Documentation Updates:**
- Final comprehensive documentation pass
- Update all affected documentation files
- Create migration notes if needed

**Commit Message Format:**
```
Phase 5: Complete multi-architecture integration

- Add cross-architecture validation suite
- Implement performance benchmarking
- Update evaluation framework for bidirectional models
- Complete documentation updates
```

---

## Development Workflow Per Phase

### Step 1: Implementation
1. Read relevant planning documents
2. Review existing code to understand integration points
3. Implement components incrementally
4. Run manual validation as you go

### Step 2: Testing
1. Write unit tests for new components
2. Write integration tests for component interactions
3. Run test suite: `pytest tests/ -v`
4. Ensure coverage meets thresholds: `pytest --cov=model_foundry`
5. Fix any failing tests before proceeding

### Step 3: Documentation
1. Identify documentation files affected by changes
2. Update inline code documentation (docstrings)
3. Update relevant markdown documentation
4. Ensure examples are up-to-date

### Step 4: Commit
1. Stage all changes: implementation + tests + docs
2. Write clear, descriptive commit message (see format above)
3. Commit to repository
4. Confirm commit succeeded

### Step 5: Communication
After each phase completion, report to project manager:
- What was implemented
- What was tested
- What was documented
- Any deviations from plan (with rationale)
- Any issues encountered
- Ready to proceed to next phase (yes/no)

---

## Code Quality Standards

### File Organization
```
model_foundry/
├── architectures/          # NEW: Architecture implementations
│   ├── __init__.py        # Registry and factory
│   ├── base.py            # Abstract base classes
│   ├── gpt.py             # GPT-2 wrapper
│   ├── bert.py            # BERT wrapper
│   ├── rnn.py             # RNN/LSTM/GRU implementations
│   └── utils.py           # Shared utilities
├── data_collators.py      # NEW: LM objective collators
├── tokenizer/
│   └── tokenizer_factory.py  # NEW: Multi-tokenizer support
├── config.py              # MODIFIED: Multi-architecture config
├── model.py               # MODIFIED: Use factory
├── data.py                # MODIFIED: Use collators
└── trainer.py             # MINOR MODIFICATIONS
```

### Naming Conventions
- Architecture classes: `{Architecture}Model` (e.g., `BERTModel`, `LSTMModel`)
- Data collators: `{Objective}DataCollator` (e.g., `MaskedLMDataCollator`)
- Config classes: `{Architecture}Config` or `{Component}Config`
- Test files: `test_{module_name}.py`
- Test functions: `test_{feature}__{scenario}` (double underscore separator)

### Code Style
- Follow existing codebase style
- Use type hints for function signatures
- Write clear docstrings (Google style)
- Keep functions focused and small
- Use descriptive variable names

### Testing Style
- Arrange-Act-Assert pattern
- One assertion concept per test (multiple asserts OK if testing same concept)
- Use fixtures for common setup
- Parameterize tests for multiple scenarios
- Mock external dependencies, use real objects for internal components

---

## Configuration Management

### Required Fields (All Configs)
```yaml
experiment_name: str          # Unique identifier
data: DataConfig             # Data pipeline configuration
tokenizer: TokenizerConfig   # Tokenizer configuration
model: ModelConfig           # Model architecture configuration
training: TrainingConfig     # Training hyperparameters
logging: LoggingConfig       # Logging configuration
random_seed: int             # Reproducibility
```

### New Model Config Structure
```yaml
model:
  architecture: "gpt2" | "bert" | "lstm" | "gru" | "rnn"  # REQUIRED

  # For transformer-based models (gpt2, bert)
  transformer:
    layers: int
    embedding_size: int
    hidden_size: int
    intermediate_hidden_size: int
    attention_heads: int
    activation_function: str
    dropout: float
    attention_dropout: float

  # BERT-specific (optional, only with architecture: bert)
  bert:
    type_vocab_size: int

  # For RNN-based models (lstm, gru, rnn)
  rnn:
    embedding_size: int
    hidden_size: int
    num_layers: int
    bidirectional: bool
    dropout: float
    rnn_type: "lstm" | "gru" | "rnn"
```

### New Training Config Additions
```yaml
training:
  # ... existing fields ...

  objective: "causal_lm" | "masked_lm"  # Training objective
  mlm_probability: float                 # For masked_lm only (default: 0.15)
```

### New Tokenizer Config Additions
```yaml
tokenizer:
  # ... existing fields ...

  tokenizer_type: "sentencepiece" | "wordpiece" | "bpe" | "character"
  special_tokens:                        # Architecture-specific tokens
    # GPT-2 style
    bos_token: "<s>"
    eos_token: "</s>"
    unk_token: "<unk>"
    pad_token: "<pad>"
    # OR BERT style
    cls_token: "[CLS]"
    sep_token: "[SEP]"
    mask_token: "[MASK]"
    unk_token: "[UNK]"
    pad_token: "[PAD]"
```

---

## Testing Configuration

### Test Dataset Requirements
- Create minimal test datasets in `tests/fixtures/datasets/`
- Size: ~100 sequences, ~50 tokens each
- Format: Tokenized HuggingFace Dataset format
- Include edge cases: empty sequences, single token, max length

### Test Model Configurations
- Tiny models for fast testing:
  - Transformer: 2 layers, 64 hidden, 2 heads
  - LSTM: 2 layers, 64 hidden
- Test configs in `tests/fixtures/configs/`
- Each architecture needs test config

### Test Execution
```bash
# Fast unit tests (during development)
pytest tests/unit/ -v

# Specific module
pytest tests/unit/test_architectures.py -v

# With coverage
pytest tests/unit/ --cov=model_foundry --cov-report=term

# Integration tests (before commit)
pytest tests/integration/ -v

# Full suite (before phase completion)
pytest tests/ -v --cov=model_foundry --cov-report=html
```

---

## Communication Protocol

### When to Ask Project Manager

**Always ask before:**
- Deviating from planned architecture
- Adding new dependencies
- Making breaking API changes
- Removing existing functionality
- Changing file structure significantly

**Report without asking:**
- Implementation progress updates
- Test results
- Documentation updates
- Bug fixes
- Code quality improvements

### Communication Format

**Progress Updates:**
```
Phase [N] Update: [Component Name]

Status: [In Progress / Completed / Blocked]

Completed:
- [List completed items]

In Progress:
- [List current work]

Tests:
- [Test status and coverage]

Issues:
- [Any issues encountered and resolution]

Next: [What's next]
```

**Phase Completion Report:**
```
Phase [N] Complete: [Phase Name]

Implemented:
- [List all implemented components]

Tested:
- [Test coverage summary]
- [Key test results]

Documented:
- [Updated documentation files]

Commit: [commit hash]

Deviations from Plan:
- [Any changes made and rationale]

Ready for Phase [N+1]: Yes/No
```

---

## Error Handling Standards

### Configuration Errors
- Fail fast with clear, actionable error messages
- Specify exactly what's wrong and how to fix it
- Example: "Missing required field 'architecture' in model config. Must be one of: gpt2, bert, lstm, gru, rnn"

### Training Errors
- Catch and log training failures
- Save checkpoint before crashing when possible
- Provide recovery instructions

### Validation Errors
- Validate configurations at load time
- Validate model/tokenizer compatibility
- Validate data format before training

---

## Performance Considerations

### Memory Management
- Different architectures have different memory requirements
- Adjust batch sizes in test configs appropriately:
  - Transformers: Smaller batches (self-attention is O(n²))
  - LSTMs: Larger batches (sequential processing)

### Training Speed
- Unit tests must complete in < 1 minute total
- Integration tests must complete in < 5 minutes total
- E2E tests can take up to 10 minutes

### Resource Usage
- All tests must run on CPU (no GPU required)
- Memory limit: 8GB for full test suite
- Disk limit: 1GB for test artifacts

---

## Version Control Practices

### Commit Frequency
- Commit after each completed, tested phase component
- Do not commit broken code
- Do not commit untested code
- Do not commit without updating relevant docs

### Commit Message Structure
```
[Phase N]: [Brief description]

[Detailed description of changes]

Components:
- [Component 1]
- [Component 2]

Tests:
- [Test coverage summary]

Docs:
- [Documentation updates]

[Optional: Breaking changes, migration notes, etc.]
```

### Branch Strategy
- Work on main branch (as per project setup)
- Each commit should be potentially releasable
- All tests must pass before commit

---

## Success Criteria

### Per-Phase Success
- [ ] All planned components implemented
- [ ] All tests passing
- [ ] Coverage meets thresholds
- [ ] Documentation updated
- [ ] Code reviewed (self-review checklist)
- [ ] Committed to repository
- [ ] PM notified of completion

### Per-Component Success
- [ ] Implements planned interface
- [ ] Has comprehensive tests
- [ ] Passes all tests
- [ ] Has documentation
- [ ] Follows code style
- [ ] No TODO comments (or documented why)

### Overall Project Success
- [ ] All architectures implemented (GPT-2, BERT, LSTM, GRU, RNN)
- [ ] All tokenizers implemented (SentencePiece, WordPiece, BPE)
- [ ] Both training objectives supported (Causal LM, Masked LM)
- [ ] Test coverage > 85% overall
- [ ] All integration tests passing
- [ ] Documentation comprehensive and accurate
- [ ] Example configs for each architecture
- [ ] Can train and evaluate all architectures

---

## Current Status Tracking

**Current Phase:** Not Started

**Completed Phases:** None

**Next Action:** Begin Phase 1 - Architecture Abstraction Layer

---

## References

- [Multi-Architecture Expansion Plan](./MULTI_ARCHITECTURE_EXPANSION.md) - Full technical specification
- [Testing Strategy](../docs/model_foundry/testing/strategy.md) - Testing requirements and standards
- [Project Structure](../docs/STRUCTURE.md) - Codebase organization (if exists)

---

## Appendix: Quick Reference Commands

### Development Commands
```bash
# Run tests
pytest tests/unit/ -v                           # Unit tests only
pytest tests/ -v                                # All tests
pytest tests/ --cov=model_foundry              # With coverage

# Check coverage
pytest --cov=model_foundry --cov-report=html   # HTML report
pytest --cov=model_foundry --cov-report=term   # Terminal report

# Run specific test
pytest tests/unit/test_architectures.py::test_gpt2_creation -v

# Check code style (if configured)
black model_foundry/ --check
ruff model_foundry/
```

### Git Commands
```bash
# Check status
git status

# Stage changes
git add model_foundry/architectures/
git add tests/unit/test_architectures.py
git add docs/model_foundry/architecture/

# Commit
git commit -m "Phase 1: Implement architecture abstraction layer"

# Verify commit
git log -1
git show HEAD
```

### Testing Individual Components
```bash
# Test config validation
pytest tests/unit/test_config.py -v

# Test model creation
pytest tests/unit/test_model.py -v

# Test architecture registry
pytest tests/unit/test_architectures.py -v

# Test data collators
pytest tests/unit/test_data_collators.py -v

# Test tokenizer factory
pytest tests/unit/test_tokenizer_factory.py -v
```

---

**End of Development Methodology Document**
