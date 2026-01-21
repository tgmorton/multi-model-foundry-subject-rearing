# Developer Guide: Custom Ablations

Learn how to add your own linguistic ablations to the preprocessing pipeline.

## Overview

The preprocessing system uses a registry pattern to discover and execute ablations. Adding a new ablation involves:

1. Implement an ablation function
2. Implement a validation function
3. Register both with the registry
4. Test your implementation

Most custom ablations can be implemented in under 30 minutes using the provided template.

## Quick Start

Copy the template and modify:

```bash
cp preprocessing/ablations/template.py preprocessing/ablations/my_ablation.py
```

Edit the file to implement your logic, then import it to trigger registration:

```python
# preprocessing/ablations/__init__.py
from . import my_ablation  # Triggers registration
```

## Ablation Function Interface

Your ablation function receives a spaCy Doc and returns modified text with a count:

```python
def my_ablation_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """
    Apply transformation to document.

    Args:
        doc: spaCy Doc object with linguistic annotations

    Returns:
        (ablated_text, num_modifications)
    """
    modified_parts = []
    num_modified = 0

    for token in doc:
        if should_modify(token):
            modified_parts.append(transform(token) + token.whitespace_)
            num_modified += 1
        else:
            modified_parts.append(token.text_with_ws)

    return ''.join(modified_parts), num_modified
```

**Key points:**
- Use `token.text_with_ws` to preserve whitespace
- Return count of modifications
- Don't modify the Doc object (it's immutable)
- Handle empty documents gracefully

## Validation Function Interface

The validator checks that your ablation worked:

```python
def validate_my_ablation(original: str, ablated: str, nlp) -> bool:
    """
    Validate that ablation occurred correctly.

    Args:
        original: Original text
        ablated: Text after ablation
        nlp: spaCy pipeline for analysis

    Returns:
        True if ablation succeeded, False otherwise
    """
    original_doc = nlp(original)
    ablated_doc = nlp(ablated)

    # Count target items
    original_count = sum(1 for token in original_doc if is_target(token))
    ablated_count = sum(1 for token in ablated_doc if is_target(token))

    # Should have fewer (or zero if none existed)
    return ablated_count < original_count if original_count > 0 else True
```

## Complete Example: Remove Adjectives

```python
"""
Remove Adjectives - Removes all adjectives from text

Tests how models learn without adjectival modification.
"""

from typing import Tuple
import spacy
from preprocessing.registry import AblationRegistry


def remove_adjectives_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """Remove all adjectives (POS tag 'ADJ')."""
    modified_parts = []
    num_removed = 0

    for token in doc:
        if token.pos_ == "ADJ":
            num_removed += 1
            # Skip this token (don't append to modified_parts)
        else:
            modified_parts.append(token.text_with_ws)

    return ''.join(modified_parts), num_removed


def validate_adjective_removal(original: str, ablated: str, nlp) -> bool:
    """Validate adjectives were removed."""
    original_doc = nlp(original)
    ablated_doc = nlp(ablated)

    original_adj = sum(1 for token in original_doc if token.pos_ == "ADJ")
    ablated_adj = sum(1 for token in ablated_doc if token.pos_ == "ADJ")

    return ablated_adj < original_adj if original_adj > 0 else True


# Register with the system
AblationRegistry.register(
    "remove_adjectives",
    remove_adjectives_doc,
    validate_adjective_removal
)
```

Use it:

```python
config = AblationConfig(
    type="remove_adjectives",  # Your ablation name
    input_path="data/raw/",
    output_path="data/processed/"
)
```

## Real-World Examples

### Lowercase All Text

```python
def lowercase_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """Convert all text to lowercase."""
    modified_parts = []
    num_modified = 0

    for token in doc:
        if token.text != token.lower_:
            modified_parts.append(token.lower_ + token.whitespace_)
            num_modified += 1
        else:
            modified_parts.append(token.text_with_ws)

    return ''.join(modified_parts), num_modified


def validate_lowercase(original: str, ablated: str, nlp) -> bool:
    """Validate text is lowercase."""
    return ablated == ablated.lower()


AblationRegistry.register(
    "lowercase",
    lowercase_doc,
    validate_lowercase
)
```

### Replace Names with Placeholder

```python
def anonymize_names_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """Replace proper nouns with [NAME]."""
    modified_parts = []
    num_replaced = 0

    for token in doc:
        if token.pos_ == "PROPN":
            modified_parts.append("[NAME]" + token.whitespace_)
            num_replaced += 1
        else:
            modified_parts.append(token.text_with_ws)

    return ''.join(modified_parts), num_replaced


def validate_anonymization(original: str, ablated: str, nlp) -> bool:
    """Validate proper nouns were replaced."""
    ablated_doc = nlp(ablated)
    # Proper nouns should be reduced (they became [NAME])
    propn_count = sum(1 for token in ablated_doc if token.pos_ == "PROPN")

    # Should have fewer proper nouns than original
    original_propn = sum(1 for token in nlp(original) if token.pos_ == "PROPN")
    return propn_count < original_propn if original_propn > 0 else True


AblationRegistry.register(
    "anonymize_names",
    anonymize_names_doc,
    validate_anonymization
)
```

## Advanced Patterns

### Context-Aware Ablation

Access surrounding tokens:

```python
def context_aware_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """Remove adjectives only after 'very'."""
    modified_parts = []
    num_removed = 0

    for i, token in enumerate(doc):
        prev_token = doc[i-1] if i > 0 else None

        # Remove adjectives after "very"
        if (prev_token and prev_token.text.lower() == "very"
            and token.pos_ == "ADJ"):
            num_removed += 1
        else:
            modified_parts.append(token.text_with_ws)

    return ''.join(modified_parts), num_removed
```

### Multi-Condition Ablation

```python
def complex_ablation_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """Remove tokens matching multiple conditions."""
    modified_parts = []
    num_removed = 0

    for token in doc:
        should_remove = (
            token.pos_ == "ADJ" or
            (token.pos_ == "ADV" and token.dep_ == "advmod") or
            token.is_stop
        )

        if should_remove:
            num_removed += 1
        else:
            modified_parts.append(token.text_with_ws)

    return ''.join(modified_parts), num_removed
```

### Factory Pattern for Runtime Configuration

For ablations that need runtime parameters:

```python
def make_remove_by_pos(pos_tags: list) -> Callable:
    """
    Create ablation function that removes specified POS tags.

    Args:
        pos_tags: List of POS tags to remove (e.g., ["ADJ", "ADV"])
    """
    def remove_by_pos_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
        modified_parts = []
        num_removed = 0

        for token in doc:
            if token.pos_ in pos_tags:
                num_removed += 1
            else:
                modified_parts.append(token.text_with_ws)

        return ''.join(modified_parts), num_removed

    return remove_by_pos_doc


# Usage
ablate_modifiers = make_remove_by_pos(["ADJ", "ADV"])
AblationRegistry.register("remove_modifiers", ablate_modifiers, validator_fn)
```

## Common Pitfalls

### Don't Forget Whitespace

```python
# WRONG - loses whitespace
modified_parts.append(token.text)

# RIGHT - preserves whitespace
modified_parts.append(token.text_with_ws)

# ALSO RIGHT - for modified tokens
modified_parts.append("modified" + token.whitespace_)
```

### Don't Modify the Doc

```python
# WRONG - Doc is immutable
for token in doc:
    token.text = "modified"

# RIGHT - Build new text
modified_parts.append("modified" + token.whitespace_)
```

### Don't Use Global State

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

## Testing Your Ablation

Create a test file:

```python
# preprocessing/tests/test_my_ablation.py

import pytest
import spacy
from preprocessing.registry import AblationRegistry


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


class TestMyAblation:

    def test_is_registered(self):
        """Ablation should be registered."""
        assert AblationRegistry.is_registered("my_ablation")

    def test_modifies_target(self, nlp):
        """Should modify target items."""
        text = "Your test text with target items"
        doc = nlp(text)
        ablation_fn, _ = AblationRegistry.get("my_ablation")

        ablated_text, num_modified = ablation_fn(doc)

        assert num_modified > 0
        assert ablated_text != text

    def test_handles_empty(self, nlp):
        """Should handle empty documents."""
        doc = nlp("")
        ablation_fn, _ = AblationRegistry.get("my_ablation")

        ablated_text, num_modified = ablation_fn(doc)

        assert ablated_text == ""
        assert num_modified == 0
```

Run tests:

```bash
python -m pytest preprocessing/tests/test_my_ablation.py -v
```

## Debugging

### Test with Small Examples

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Test sentence here")

ablated, count = my_ablation_doc(doc)
print(f"Original: '{doc.text}'")
print(f"Ablated:  '{ablated}'")
print(f"Modified: {count} items")
```

### Inspect Token Attributes

```python
for token in doc:
    print(f"{token.text:15} POS={token.pos_:5} DEP={token.dep_:10}")
```

### Enable Verbose Logging

```python
config = AblationConfig(
    type="my_ablation",
    input_path="data/test/",
    output_path="data/output/",
    verbose=True  # Detailed logs
)
```

## Performance Considerations

### Disable Unused spaCy Components

```python
config = AblationConfig(
    type="my_ablation",
    input_path="data/corpus/",
    output_path="data/processed/",
    # Only enable what you need
    spacy_disable_components=["ner", "textcat", "lemmatizer"]
)
```

Common needs by component:
- **tagger**: POS tags (`token.pos_`)
- **parser**: Dependencies (`token.dep_`, `token.head`)
- **ner**: Named entities
- **lemmatizer**: Lemmas (`token.lemma_`)

### Optimize Batch Size

```python
config = AblationConfig(
    type="my_ablation",
    input_path="data/corpus/",
    output_path="data/processed/",
    spacy_batch_size=100  # Larger = faster (more memory)
)
```

## Integration

Add to ablations package:

```python
# preprocessing/ablations/__init__.py
from . import remove_articles
from . import remove_expletives
from . import impoverish_determiners
from . import lemmatize_verbs
from . import remove_subject_pronominals
from . import my_ablation  # Add your module

__all__ = [
    "remove_articles",
    "remove_expletives",
    "impoverish_determiners",
    "lemmatize_verbs",
    "remove_subject_pronominals",
    "my_ablation",
]
```

Now it's available:

```python
config = AblationConfig(
    type="my_ablation",
    input_path="data/raw/",
    output_path="data/processed/"
)
```

## Next Steps

**Need more examples?** Check existing ablations in `preprocessing/ablations/`

**Want advanced features?** See [Advanced Usage](ADVANCED_USAGE.md) for factory patterns and coreference resolution

**Ready to contribute?** See [Testing Guide](TESTING.md) for test requirements
