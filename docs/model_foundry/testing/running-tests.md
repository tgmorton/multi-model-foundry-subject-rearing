# Model Foundry Test Suite

Comprehensive test suite for the Model Foundry framework.

## Quick Start

### Install Test Dependencies

```bash
pip install pytest pytest-cov pytest-mock pytest-timeout
```

### Run All Tests

```bash
# From project root
pytest model_foundry/tests/

# Or using the configured test path
pytest
```

## Test Organization

```
tests/
├── unit/                    # Fast unit tests (< 0.1s each)
│   ├── test_config.py       # Configuration validation
│   ├── test_data.py         # Data processing
│   ├── test_model.py        # Model creation
│   ├── test_utils.py        # Utility functions
│   └── training/
│       ├── test_tokenization.py
│       ├── test_checkpointing.py  # Critical: checkpoint reliability
│       └── test_loop.py
├── integration/             # Integration tests (< 5s each)
│   ├── test_data_pipeline.py
│   ├── test_training_pipeline.py
│   └── test_checkpoint_recovery.py
├── e2e/                     # End-to-end tests (< 60s each)
│   └── test_full_training_run.py
├── fixtures/                # Test data and configs
└── conftest.py              # Shared fixtures and configuration
```

## Running Tests

### By Category

```bash
# Unit tests only (fastest)
pytest model_foundry/tests/unit/ -v

# Integration tests
pytest model_foundry/tests/integration/ -v

# End-to-end tests
pytest model_foundry/tests/e2e/ -v
```

### By Module

```bash
# Test configuration validation
pytest model_foundry/tests/unit/test_config.py -v

# Test checkpointing (critical)
pytest model_foundry/tests/unit/training/test_checkpointing.py -v

# Test data processing
pytest model_foundry/tests/unit/test_data.py -v
```

### By Test Function

```bash
# Run a specific test
pytest model_foundry/tests/unit/test_config.py::TestDataConfig::test_valid_data_config -v
```

### Using Markers

```bash
# Skip slow tests
pytest -m "not slow"

# Run only GPU tests
pytest -m gpu

# Run only integration tests
pytest -m integration

# Combine markers
pytest -m "unit and not slow"
```

## Coverage

### Generate Coverage Report

```bash
# HTML report (opens in browser)
pytest --cov=model_foundry --cov-report=html
open htmlcov/index.html

# Terminal report
pytest --cov=model_foundry --cov-report=term-missing

# XML report (for CI/CD)
pytest --cov=model_foundry --cov-report=xml
```

### Coverage Goals

| Component | Target | Status |
|-----------|--------|--------|
| config.py | 95%+ | ✅ |
| data.py | 90%+ | 🟡 |
| training/checkpointing.py | 90%+ | ✅ |
| training/loop.py | 85%+ | 🟡 |
| training/tokenization.py | 85%+ | 🟡 |
| trainer.py | 80%+ | 🟡 |
| Overall | 85%+ | 🟡 |

## Test Development

### Writing New Tests

1. **Choose the right location:**
   - `unit/` for isolated component tests
   - `integration/` for multi-component tests
   - `e2e/` for full pipeline tests

2. **Use fixtures from `conftest.py`:**
   ```python
   def test_something(tiny_config, temp_workspace):
       # tiny_config provides a minimal test configuration
       # temp_workspace provides a clean temporary directory
       pass
   ```

3. **Follow naming conventions:**
   - Test files: `test_*.py`
   - Test classes: `Test*`
   - Test functions: `test_*`

4. **Add markers for categorization:**
   ```python
   @pytest.mark.slow
   @pytest.mark.gpu
   def test_training_on_gpu():
       pass
   ```

### Common Fixtures

Available in `conftest.py`:

- `tiny_config` - Minimal valid configuration
- `tiny_model` - Small GPT-2 model for testing
- `tiny_dataset` - Small tokenized dataset
- `mock_tokenizer` - Mock tokenizer (no dependencies)
- `temp_workspace` - Clean temporary workspace
- `device` - CPU or CUDA device
- `deterministic_seed` - Set seeds for reproducibility

### Test Best Practices

1. **Keep tests fast:**
   - Unit tests < 0.1s
   - Integration tests < 5s
   - E2E tests < 60s

2. **Make tests independent:**
   - Each test should be runnable in isolation
   - Use fixtures for setup/teardown
   - Don't rely on test execution order

3. **Use descriptive names:**
   ```python
   # Good
   def test_checkpoint_saves_optimizer_state():
       pass

   # Bad
   def test_checkpoint():
       pass
   ```

4. **Test edge cases:**
   - Empty inputs
   - Maximum/minimum values
   - Invalid inputs
   - Boundary conditions

5. **Use parametrize for multiple scenarios:**
   ```python
   @pytest.mark.parametrize("batch_size,seq_len", [
       (1, 32),
       (16, 128),
       (32, 512),
   ])
   def test_data_loading(batch_size, seq_len):
       pass
   ```

## Continuous Integration

### GitHub Actions

Tests run automatically on:
- Every push to main
- Every pull request
- Nightly builds

### Local Pre-commit

Run tests before committing:

```bash
# Fast checks only
pytest model_foundry/tests/unit/ -x

# Full test suite
pytest model_foundry/tests/ -x
```

Add to `.git/hooks/pre-commit`:
```bash
#!/bin/bash
pytest model_foundry/tests/unit/ -x --tb=short
```

## Debugging Tests

### Run with More Verbose Output

```bash
# Show print statements
pytest -s

# Show full tracebacks
pytest --tb=long

# Stop at first failure
pytest -x

# Show local variables in tracebacks
pytest -l
```

### Run in Debugger

```bash
# Drop into pdb on failure
pytest --pdb

# Drop into pdb on first test
pytest --trace
```

### Profile Slow Tests

```bash
# Show slowest 10 tests
pytest --durations=10

# Show all test durations
pytest --durations=0
```

## Common Issues

### Import Errors

If you get import errors, make sure model_foundry is installed:

```bash
pip install -e .
```

### CUDA Out of Memory

Skip GPU tests or reduce batch size:

```bash
pytest -m "not gpu"
```

### Flaky Tests

Run multiple times to identify flaky tests:

```bash
pytest --count=10 model_foundry/tests/unit/test_specific.py
```

## Test Maintenance

### Weekly Tasks

- [ ] Review test coverage report
- [ ] Check for flaky tests
- [ ] Update fixtures if needed

### Monthly Tasks

- [ ] Review and update this README
- [ ] Clean up deprecated tests
- [ ] Add tests for new features

### Before Release

- [ ] Run full test suite
- [ ] Run E2E tests
- [ ] Check coverage > 85%
- [ ] Verify all critical tests pass

## Contributing

When adding new features:

1. Write tests first (TDD)
2. Ensure tests pass locally
3. Add markers if needed
4. Update this README if necessary
5. Submit PR with tests included

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [pytest markers](https://docs.pytest.org/en/stable/mark.html)
- [Testing best practices](https://docs.pytest.org/en/stable/goodpractices.html)

## Support

For test-related questions:
- Open an issue on GitHub
- Tag with `testing` label
- Include minimal reproducible example
