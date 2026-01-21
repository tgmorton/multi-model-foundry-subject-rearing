# Developer Guide: Adding Custom Ablations

This guide shows you how to add a new ablation to the preprocessing pipeline in under 30 minutes.

## Quick Start Checklist

- [ ] Copy the ablation template
- [ ] Implement the ablation function
- [ ] Implement the validation function
- [ ] Register the ablation
- [ ] Write tests
- [ ] Test your ablation
- [ ] Use it in a pipeline

**Estimated time**: 15-30 minutes for a simple ablation

## Step 1: Copy the Template

```bash
cp preprocessing/ablations/template.py preprocessing/ablations/my_ablation.py
```

Or create from scratch using the template below.

## Step 2: Implement Your Ablation

### Basic Template

```python
"""
<Ablation Name> - Brief description

Detailed description of what this ablation does and why it's useful.
"""

from typing import Tuple
import spacy
from preprocessing.registry import AblationRegistry


def my_ablation_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """
    <One-line description of the transformation>

    Args:
        doc: spaCy Doc object to process

    Returns:
        Tuple of (ablated_text, num_modifications)
    """
    modified_parts = []
    num_modifications = 0

    for token in doc:
        # Your ablation logic here
        if <condition_to_modify>:
            # Modify the token
            modified_parts.append(<modified_token> + token.whitespace_)
            num_modifications += 1
        else:
            # Keep original
            modified_parts.append(token.text_with_ws)

    return ''.join(modified_parts), num_modifications


def validate_my_ablation(original: str, ablated: str, nlp) -> bool:
    """
    Validate that the ablation occurred.

    Args:
        original: Original text before ablation
        ablated: Text after ablation
        nlp: spaCy NLP pipeline

    Returns:
        True if ablation was successful, False otherwise
    """
    original_doc = nlp(original)
    ablated_doc = nlp(ablated)

    # Count relevant items in original
    original_count = sum(1 for token in original_doc if <condition>)

    # Count relevant items in ablated
    ablated_count = sum(1 for token in ablated_doc if <condition>)

    # Should be fewer (or zero if none existed)
    return ablated_count < original_count if original_count > 0 else True


# Register the ablation
AblationRegistry.register(
    "my_ablation",
    my_ablation_doc,
    validate_my_ablation
)
```

## Step 3: Real Examples

### Example 1: Remove Adjectives

```python
"""
Remove all adjectives from text.

This ablation tests how models learn without adjectival modification.
"""

from typing import Tuple
import spacy
from preprocessing.registry import AblationRegistry


def remove_adjectives_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """
    Remove all adjectives (POS tag 'ADJ') from text.

    Args:
        doc: spaCy Doc object to process

    Returns:
        Tuple of (ablated_text, num_removed)
    """
    modified_parts = []
    num_removed = 0

    for token in doc:
        if token.pos_ == "ADJ":
            # Skip adjectives (don't add to modified_parts)
            num_removed += 1
        else:
            # Keep everything else
            modified_parts.append(token.text_with_ws)

    return ''.join(modified_parts), num_removed


def validate_adjective_removal(original: str, ablated: str, nlp) -> bool:
    """
    Validate that adjectives were removed.

    Args:
        original: Original text
        ablated: Ablated text
        nlp: spaCy pipeline

    Returns:
        True if adjectives were reduced or none existed
    """
    original_doc = nlp(original)
    ablated_doc = nlp(ablated)

    original_adj = sum(1 for token in original_doc if token.pos_ == "ADJ")
    ablated_adj = sum(1 for token in ablated_doc if token.pos_ == "ADJ")

    return ablated_adj < original_adj if original_adj > 0 else True


# Register
AblationRegistry.register(
    "remove_adjectives",
    remove_adjectives_doc,
    validate_adjective_removal
)
```

### Example 2: Lowercase All Text

```python
"""
Convert all text to lowercase.

Tests case-insensitive learning.
"""

from typing import Tuple
import spacy
from preprocessing.registry import AblationRegistry


def lowercase_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """
    Convert all tokens to lowercase.

    Args:
        doc: spaCy Doc object to process

    Returns:
        Tuple of (ablated_text, num_modified)
    """
    modified_parts = []
    num_modified = 0

    for token in doc:
        if token.text != token.lower_:
            # Token needs lowercasing
            modified_parts.append(token.lower_ + token.whitespace_)
            num_modified += 1
        else:
            # Already lowercase
            modified_parts.append(token.text_with_ws)

    return ''.join(modified_parts), num_modified


def validate_lowercase(original: str, ablated: str, nlp) -> bool:
    """
    Validate that text was lowercased.

    Args:
        original: Original text
        ablated: Ablated text
        nlp: spaCy pipeline

    Returns:
        True if text is now lowercase
    """
    # Simple check: ablated should equal ablated.lower()
    return ablated == ablated.lower()


# Register
AblationRegistry.register(
    "lowercase",
    lowercase_doc,
    validate_lowercase
)
```

### Example 3: Replace with Placeholder

```python
"""
Replace all proper nouns with [NAME].

Tests model behavior without specific names.
"""

from typing import Tuple
import spacy
from preprocessing.registry import AblationRegistry


def anonymize_names_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """
    Replace all proper nouns (PROPN) with [NAME].

    Args:
        doc: spaCy Doc object to process

    Returns:
        Tuple of (ablated_text, num_replaced)
    """
    modified_parts = []
    num_replaced = 0

    for token in doc:
        if token.pos_ == "PROPN":
            # Replace with placeholder
            modified_parts.append("[NAME]" + token.whitespace_)
            num_replaced += 1
        else:
            modified_parts.append(token.text_with_ws)

    return ''.join(modified_parts), num_replaced


def validate_anonymization(original: str, ablated: str, nlp) -> bool:
    """
    Validate that proper nouns were replaced.

    Args:
        original: Original text
        ablated: Ablated text
        nlp: spaCy pipeline

    Returns:
        True if proper nouns were replaced or none existed
    """
    original_doc = nlp(original)
    ablated_doc = nlp(ablated)

    original_propn = sum(1 for token in original_doc if token.pos_ == "PROPN")
    ablated_propn = sum(1 for token in ablated_doc if token.pos_ == "PROPN")

    # Should have fewer proper nouns (they became [NAME])
    if original_propn > 0:
        return ablated_propn < original_propn
    return True


# Register
AblationRegistry.register(
    "anonymize_names",
    anonymize_names_doc,
    validate_anonymization
)
```

## Step 4: Add to Ablations Package

Update `preprocessing/ablations/__init__.py`:

```python
# Import all ablation modules to trigger registration
from . import remove_articles
from . import remove_expletives
from . import impoverish_determiners
from . import lemmatize_verbs
from . import remove_subject_pronominals
from . import my_ablation  # ADD YOUR MODULE HERE

__all__ = [
    "remove_articles",
    "remove_expletives",
    "impoverish_determiners",
    "lemmatize_verbs",
    "remove_subject_pronominals",
    "my_ablation",  # AND HERE
]
```

## Step 5: Write Tests

Create `preprocessing/tests/test_my_ablation_integration.py`:

```python
"""
Integration tests for my_ablation.
"""

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
    """Tests for my_ablation registration."""

    def test_is_registered(self):
        """my_ablation should be registered."""
        assert AblationRegistry.is_registered("my_ablation")

    def test_can_retrieve(self):
        """Should be able to retrieve my_ablation function."""
        ablation_fn, validator_fn = AblationRegistry.get("my_ablation")
        assert callable(ablation_fn)
        assert callable(validator_fn)


class TestMyAblationFunction:
    """Tests for my_ablation function."""

    def test_modifies_target(self, nlp):
        """Should modify target items."""
        text = "Your test text here"
        doc = nlp(text)
        ablation_fn, _ = AblationRegistry.get("my_ablation")

        ablated_text, num_modified = ablation_fn(doc)

        assert num_modified > 0  # Should have modified something
        assert ablated_text != text  # Should be different

    def test_preserves_non_targets(self, nlp):
        """Should preserve non-target items."""
        text = "Text with items you want to keep"
        doc = nlp(text)
        ablation_fn, _ = AblationRegistry.get("my_ablation")

        ablated_text, num_modified = ablation_fn(doc)

        assert "items you want" in ablated_text  # Should preserve these

    def test_handles_empty_doc(self, nlp):
        """Should handle empty documents."""
        text = ""
        doc = nlp(text)
        ablation_fn, _ = AblationRegistry.get("my_ablation")

        ablated_text, num_modified = ablation_fn(doc)

        assert ablated_text == ""
        assert num_modified == 0
```

## Step 6: Test Your Ablation

```bash
# Run your tests
python -m pytest preprocessing/tests/test_my_ablation_integration.py -v

# Run all tests to ensure nothing broke
python -m pytest preprocessing/tests/ -v
```

## Step 7: Use It

```python
from preprocessing.config import AblationConfig
from preprocessing.base import AblationPipeline

config = AblationConfig(
    type="my_ablation",  # Your ablation name
    input_path="data/raw/corpus/",
    output_path="data/processed/my_ablation/",
    seed=42
)

pipeline = AblationPipeline(config)
manifest = pipeline.process_corpus()

print(f"Modified {manifest.metadata.total_items_ablated:,} items")
```

## Advanced Patterns

### Pattern 1: Multi-Condition Ablation

```python
def complex_ablation_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """Remove tokens matching multiple conditions."""
    modified_parts = []
    num_removed = 0

    for token in doc:
        # Multiple conditions
        should_remove = (
            token.pos_ == "ADJ" or
            (token.pos_ == "ADV" and token.dep_ == "advmod") or
            token.is_stop
        )

        if not should_remove:
            modified_parts.append(token.text_with_ws)
        else:
            num_removed += 1

    return ''.join(modified_parts), num_removed
```

### Pattern 2: Context-Aware Ablation

```python
def context_aware_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """Remove items based on surrounding context."""
    modified_parts = []
    num_removed = 0

    for i, token in enumerate(doc):
        # Check previous token
        prev_token = doc[i-1] if i > 0 else None

        # Check next token
        next_token = doc[i+1] if i < len(doc) - 1 else None

        # Condition based on context
        if prev_token and prev_token.text == "very" and token.pos_ == "ADJ":
            # Remove adjectives after "very"
            num_removed += 1
        else:
            modified_parts.append(token.text_with_ws)

    return ''.join(modified_parts), num_removed
```

### Pattern 3: Factory Function (Advanced)

For ablations that need runtime configuration:

```python
def make_remove_by_pos(pos_tags: List[str]):
    """
    Create an ablation function that removes specified POS tags.

    Args:
        pos_tags: List of POS tags to remove (e.g., ["ADJ", "ADV"])

    Returns:
        Ablation function
    """
    def remove_by_pos_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
        """Remove tokens with specified POS tags."""
        modified_parts = []
        num_removed = 0

        for token in doc:
            if token.pos_ in pos_tags:
                num_removed += 1
            else:
                modified_parts.append(token.text_with_ws)

        return ''.join(modified_parts), num_removed

    return remove_by_pos_doc


# Usage:
ablate_fn = make_remove_by_pos(["ADJ", "ADV"])
AblationRegistry.register("remove_modifiers", ablate_fn, validator_fn)
```

## Common Pitfalls

### ❌ Don't forget whitespace
```python
# WRONG
modified_parts.append(token.text)

# RIGHT
modified_parts.append(token.text_with_ws)
```

### ❌ Don't modify the Doc object
```python
# WRONG - Doc is immutable
for token in doc:
    token.text = "modified"

# RIGHT - Build new text
modified_parts.append("modified" + token.whitespace_)
```

### ❌ Don't use global state
```python
# WRONG - Not thread-safe
count = 0
def ablation(doc):
    global count
    count += 1

# RIGHT - Return count
def ablation(doc):
    count = 0
    # ... process ...
    return text, count
```

## Debugging Tips

### 1. Use verbose mode
```python
config = AblationConfig(
    type="my_ablation",
    input_path="data/test/",
    output_path="data/output/",
    verbose=True  # Detailed logging
)
```

### 2. Test with small examples
```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Test sentence")

ablated, count = my_ablation_doc(doc)
print(f"Result: '{ablated}'")
print(f"Modified: {count}")
```

### 3. Check token attributes
```python
for token in doc:
    print(f"{token.text:10} POS={token.pos_:5} DEP={token.dep_:10}")
```

## Performance Optimization

### Use spaCy components selectively
```python
config = AblationConfig(
    type="my_ablation",
    input_path="data/raw/",
    output_path="data/processed/",
    # Only enable what you need
    spacy_disable_components=["ner", "textcat", "lemmatizer"]
)
```

### Optimize batch size
```python
config = AblationConfig(
    type="my_ablation",
    input_path="data/raw/",
    output_path="data/processed/",
    spacy_batch_size=100  # Larger = faster (more memory)
)
```

## Next Steps

- See [User Guide](USER_GUIDE.md) for usage examples
- See [Advanced Usage](ADVANCED_USAGE.md) for complex patterns
- See [Testing Guide](TESTING.md) for test best practices

## Getting Help

Questions? Check:
1. Existing ablations in `preprocessing/ablations/`
2. [Test examples](../../preprocessing/tests/)
3. This guide
4. File an issue with your code

Happy ablating! 🎉
