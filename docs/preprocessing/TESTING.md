# Preprocessing Tests

## Test Overview

This directory contains comprehensive tests for the preprocessing pipeline.

### Test Files

1. **test_registry.py** (14 tests) - AblationRegistry functionality
2. **test_config.py** (23 tests) - Pydantic configuration models
3. **test_utils.py** (25 tests) - Utility functions
4. **test_remove_articles_integration.py** (2 tests) - Integration tests for remove_articles ablation
5. **test_base.py** (8 tests - SKIPPED) - Pipeline base class tests

**Total: 64 passing tests** (8 skipped due to numpy incompatibility)

## Running Tests

```bash
# Run all passing tests
pytest preprocessing/tests/test_registry.py \
       preprocessing/tests/test_config.py \
       preprocessing/tests/test_utils.py \
       preprocessing/tests/test_remove_articles_integration.py -v

# Run specific test file
pytest preprocessing/tests/test_registry.py -v

# Run specific test
pytest preprocessing/tests/test_registry.py::TestAblationRegistry::test_register_ablation_without_validator -v
```

## Known Issues

### NumPy Version Incompatibility

**Issue:** spaCy 3.8.7 requires numpy 2.x, but transformers requires numpy < 2.0.

**Affected Tests:** `test_base.py` (all tests skipped)

**Error:**
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
Expected 96 from C header, got 88 from PyObject
```

**Resolution Options:**
1. Wait for transformers to support numpy 2.x
2. Downgrade spaCy to version compatible with numpy < 2.0
3. Use separate virtual environments for preprocessing vs training
4. Skip pipeline integration tests until resolved

**Current Status:** Tests in `test_base.py` are marked with `pytestmark = pytest.mark.skip()`

## Test Isolation

### Registry Cleanup

The `TestAblationRegistry` class clears the registry before each test using `setup_method()` and restores it in `teardown_method()` using `importlib.reload()`. This ensures:

1. Registry tests run in isolation
2. Integration tests can still find registered ablations
3. No test pollution between test classes

### Session-Scoped Registration

A session-scoped fixture `_register_ablations` in `conftest.py` ensures ablations are registered once at the start of the test session, making them available to all integration tests.

## Fixtures

See `conftest.py` for available fixtures:

- `clean_registry` - Clears registry (opt-in)
- `sample_corpus_dir` - Creates test corpus with .train files
- `sample_pool_dir` - Creates replacement pool directory
- `dummy_ablation_function` - Mock ablation for testing
- `dummy_validator_function` - Mock validator for testing
- `mock_spacy_doc` - Mock spaCy Doc object

## Adding New Tests

### For New Ablations

1. Create `test_{ablation_name}_integration.py`
2. Test registration:
   ```python
   def test_{ablation}_is_registered(self):
       from preprocessing.ablations import {ablation}  # noqa
       assert AblationRegistry.is_registered("{ablation}")
   ```

3. Test ablation function directly with mock spaCy docs
4. Test validator function
5. Test full pipeline (if spaCy models available)

### For Core Components

- Add tests to appropriate file (test_registry.py, test_config.py, test_utils.py)
- Use `clean_registry` fixture if testing requires empty registry
- Use pytest's tmp_path fixture for file I/O tests
