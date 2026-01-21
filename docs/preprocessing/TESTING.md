# Preprocessing Tests

Guide to running and writing tests for the preprocessing pipeline.

## Running Tests

### All Tests

```bash
python -m pytest preprocessing/tests/ -v
```

### Specific Test File

```bash
python -m pytest preprocessing/tests/test_base.py -v
```

### With Coverage

```bash
python -m pytest preprocessing/tests/ --cov=preprocessing --cov-report=html

# View report
open htmlcov/index.html
```

### By Category

```bash
# Unit tests only
python -m pytest preprocessing/tests/test_*.py -v

# Integration tests
python -m pytest preprocessing/tests/test_*_integration.py -v
```

## Test Status

Current coverage: **106/106 tests passing**

Test breakdown:
- `test_base.py`: 8 tests (Pipeline initialization, configuration)
- `test_config.py`: 23 tests (Pydantic models, validation)
- `test_registry.py`: 14 tests (Registration, retrieval)
- `test_utils.py`: 24 tests (Utility functions)
- `test_remove_articles_integration.py`: 10 tests (Article removal)
- `test_new_ablations_integration.py`: 27 tests (4 new ablations)

## Writing Tests for New Ablations

### Basic Test Structure

```python
# preprocessing/tests/test_my_ablation_integration.py

import pytest
import spacy
from preprocessing.registry import AblationRegistry


@pytest.fixture(scope="module")
def nlp():
    """Load spaCy model once for all tests."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("spaCy model not available")


class TestMyAblationRegistration:
    """Test that ablation is properly registered."""

    def test_is_registered(self):
        """Ablation should be in registry."""
        assert AblationRegistry.is_registered("my_ablation")

    def test_can_retrieve(self):
        """Should retrieve ablation and validator."""
        ablation_fn, validator_fn = AblationRegistry.get("my_ablation")
        assert callable(ablation_fn)
        assert callable(validator_fn)


class TestMyAblationFunction:
    """Test ablation behavior."""

    def test_modifies_target(self, nlp):
        """Should modify target items."""
        text = "Text with target items"
        doc = nlp(text)
        ablation_fn, _ = AblationRegistry.get("my_ablation")

        ablated_text, num_modified = ablation_fn(doc)

        assert num_modified > 0
        assert ablated_text != text

    def test_preserves_non_targets(self, nlp):
        """Should preserve non-target items."""
        text = "Text with items to keep"
        doc = nlp(text)
        ablation_fn, _ = AblationRegistry.get("my_ablation")

        ablated_text, num_modified = ablation_fn(doc)

        assert "items to keep" in ablated_text

    def test_handles_empty_doc(self, nlp):
        """Should handle empty documents."""
        doc = nlp("")
        ablation_fn, _ = AblationRegistry.get("my_ablation")

        ablated_text, num_modified = ablation_fn(doc)

        assert ablated_text == ""
        assert num_modified == 0

    def test_handles_no_targets(self, nlp):
        """Should handle text with no targets."""
        text = "Text with no targets"
        doc = nlp(text)
        ablation_fn, _ = AblationRegistry.get("my_ablation")

        ablated_text, num_modified = ablation_fn(doc)

        assert num_modified == 0
        assert ablated_text == text


class TestMyAblationValidation:
    """Test validation function."""

    def test_validates_successful_ablation(self, nlp):
        """Validator should pass for successful ablation."""
        text = "Text with targets"
        doc = nlp(text)
        ablation_fn, validator_fn = AblationRegistry.get("my_ablation")

        ablated_text, _ = ablation_fn(doc)
        is_valid = validator_fn(text, ablated_text, nlp)

        assert is_valid

    def test_validator_handles_no_changes(self, nlp):
        """Validator should pass when no changes needed."""
        text = "Text with no targets"
        doc = nlp(text)
        ablation_fn, validator_fn = AblationRegistry.get("my_ablation")

        ablated_text, _ = ablation_fn(doc)
        is_valid = validator_fn(text, ablated_text, nlp)

        assert is_valid
```

### Test Fixtures

Common fixtures in `conftest.py`:

```python
@pytest.fixture
def sample_corpus_dir(tmp_path):
    """Create temporary corpus directory."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    # Create sample files
    (corpus_dir / "file1.train").write_text("Sample text here")
    (corpus_dir / "file2.train").write_text("More sample text")

    return corpus_dir


@pytest.fixture
def sample_config(sample_corpus_dir, tmp_path):
    """Create sample configuration."""
    return AblationConfig(
        type="remove_articles",
        input_path=str(sample_corpus_dir),
        output_path=str(tmp_path / "output"),
        seed=42
    )
```

## Testing Best Practices

### Test Isolation

Each test should be independent:

```python
# GOOD - Test is self-contained
def test_feature(self, nlp):
    text = "Test input"
    doc = nlp(text)
    result, count = my_ablation_doc(doc)
    assert count > 0

# BAD - Depends on external state
global_counter = 0
def test_feature(self, nlp):
    global global_counter
    # ... test depends on global state
```

### Descriptive Names

```python
# GOOD - Clear what's being tested
def test_removes_all_adjectives_from_sentence(self):
    ...

def test_handles_empty_document_without_error(self):
    ...

# BAD - Vague names
def test_ablation(self):
    ...

def test_it_works(self):
    ...
```

### Test One Thing

```python
# GOOD - Focused test
def test_removes_adjectives(self, nlp):
    doc = nlp("The big red car")
    ablated, count = remove_adjectives_doc(doc)
    assert "big" not in ablated
    assert "red" not in ablated

# BAD - Tests too many things
def test_ablation_works(self, nlp):
    # Tests removal, counting, validation, edge cases...
```

### Use Parametrize for Variants

```python
@pytest.mark.parametrize("text,expected_count", [
    ("The cat", 1),  # One article
    ("A big red cat", 1),  # One article, preserve adjectives
    ("No articles here", 0),  # No articles
    ("", 0),  # Empty
])
def test_article_counting(self, nlp, text, expected_count):
    doc = nlp(text)
    ablated, count = remove_articles_doc(doc)
    assert count == expected_count
```

## Common Test Patterns

### Testing Whitespace Preservation

```python
def test_preserves_whitespace(self, nlp):
    text = "The  cat  sat  on  a  mat"  # Multiple spaces
    doc = nlp(text)
    ablated, _ = remove_articles_doc(doc)

    # Whitespace should be preserved
    assert "  " in ablated  # Double spaces remain
```

### Testing Edge Cases

```python
def test_handles_unicode(self, nlp):
    text = "The café has a naïve owner"
    doc = nlp(text)
    ablated, count = remove_articles_doc(doc)
    assert count == 2
    assert "café" in ablated
    assert "naïve" in ablated

def test_handles_punctuation(self, nlp):
    text = "The cat! A dog? The bird."
    doc = nlp(text)
    ablated, count = remove_articles_doc(doc)
    assert "!" in ablated
    assert "?" in ablated
    assert "." in ablated
```

### Testing Validation

```python
def test_validation_detects_failure(self, nlp):
    """Validator should fail if ablation didn't work."""
    original = "The cat sat on a mat"
    failed_ablation = original  # No changes made

    _, validator_fn = AblationRegistry.get("remove_articles")
    is_valid = validator_fn(original, failed_ablation, nlp)

    assert not is_valid  # Should detect that articles weren't removed
```

## Continuous Integration

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
          pip install pytest pytest-cov spacy tqdm
          python -m spacy download en_core_web_sm

      - name: Run tests
        run: |
          pytest preprocessing/tests/ --cov=preprocessing --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Debugging Failed Tests

### Run with Verbose Output

```bash
pytest preprocessing/tests/test_my_ablation.py -v -s
```

The `-s` flag shows print statements.

### Run Single Test

```bash
pytest preprocessing/tests/test_my_ablation.py::TestMyAblation::test_specific_case -v
```

### Use pdb Debugger

```python
def test_feature(self, nlp):
    text = "Debug this"
    doc = nlp(text)

    import pdb; pdb.set_trace()  # Debugger starts here

    ablated, count = my_ablation_doc(doc)
    assert count > 0
```

### Check Test Fixtures

```python
def test_fixture_works(self, sample_corpus_dir):
    """Verify fixture creates what you expect."""
    files = list(sample_corpus_dir.glob("*.train"))
    print(f"Found files: {files}")
    assert len(files) > 0
```

## Coverage Goals

Target coverage by module:

- **ablations/**: 90%+ (core functionality)
- **base.py**: 85%+ (pipeline orchestration)
- **config.py**: 95%+ (validation logic)
- **registry.py**: 95%+ (registration system)
- **utils.py**: 90%+ (helper functions)

Check current coverage:

```bash
pytest preprocessing/tests/ --cov=preprocessing --cov-report=term-missing
```

## Next Steps

**Adding tests for your ablation**: Use the test structure template above

**Understanding test fixtures**: See `preprocessing/tests/conftest.py`

**CI/CD integration**: Add GitHub Actions workflow above to `.github/workflows/`
