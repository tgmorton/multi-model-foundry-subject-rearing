# Advanced Preprocessing Usage

## Using Coreference Resolution with remove_expletives

The `remove_expletives` ablation supports two modes:

### Simple Mode (Default)
Uses dependency parsing to identify and remove tokens marked as expletives ('expl' dependency label).

```python
from preprocessing.config import AblationConfig
from preprocessing.base import AblationPipeline

config = AblationConfig(
    type="remove_expletives",
    input_path="data/raw/corpus/",
    output_path="data/processed/no_expletives/",
    seed=42
)

pipeline = AblationPipeline(config)
manifest = pipeline.process_corpus()
```

### Advanced Mode (With Coreference Resolution)
Uses coreference resolution to confirm that pronouns are truly non-referential before removing them. This mode is more accurate, especially for long-distance dependencies.

```python
import spacy
from preprocessing.ablations.remove_expletives import make_remove_expletives_with_coref
from preprocessing.registry import AblationRegistry
from preprocessing.config import AblationConfig
from preprocessing.base import AblationPipeline

# Load a spaCy model with coreference resolution
# Option 1: Use the same model (basic coreference)
nlp_coref = spacy.load("en_core_web_sm")

# Option 2: Use a specialized coreference model (better accuracy)
# First install: pip install spacy-experimental
# Then: python -m spacy download en_coreference_web_trf
# nlp_coref = spacy.load("en_coreference_web_trf")

# Create the coreference-enabled ablation function
ablate_with_coref = make_remove_expletives_with_coref(nlp_coref)

# Temporarily register it (overwrite the simple version)
AblationRegistry.unregister("remove_expletives")
AblationRegistry.register(
    "remove_expletives",
    ablate_with_coref,
    # Use the same validator
    AblationRegistry._validators.get("remove_expletives")
)

# Now use it with the pipeline
config = AblationConfig(
    type="remove_expletives",
    input_path="data/raw/corpus/",
    output_path="data/processed/no_expletives_coref/",
    seed=42
)

pipeline = AblationPipeline(config)
manifest = pipeline.process_corpus()
```

### How Coreference Resolution Improves Accuracy

The advanced mode includes context from the previous sentence when checking for coreference:

```python
# Example text:
# "I saw a cat. It was sleeping on the porch."

# Simple mode:
# - Might incorrectly remove "It" if tagged as 'expl'

# Advanced mode with coreference:
# - Checks: "I saw a cat. It was sleeping..."
# - Finds "It" refers to "cat" in coreference chain
# - Correctly preserves "It" as referential
```

Versus true expletives:

```python
# Example text:
# "It is raining outside."

# Both modes:
# - "It" has no antecedent (non-referential)
# - Correctly removes "It"
# Result: "is raining outside."
```

### Performance Considerations

- **Simple mode**: Fast, works well for most cases
- **Advanced mode**: Slower (runs coreference resolution for each potential expletive), but more accurate

For large corpora (>100M tokens), simple mode is recommended unless high precision is critical.

### Custom Script Example

For maximum control, you can use the ablation function directly:

```python
import spacy
from preprocessing.ablations.remove_expletives import (
    make_remove_expletives_with_coref,
    remove_expletives_doc
)

# Load model
nlp = spacy.load("en_core_web_sm")

# Process text
text = "The report was late. It arrived yesterday. It was raining."

# Simple mode
doc = nlp(text)
ablated_simple, count_simple = remove_expletives_doc(doc)
print(f"Simple: {ablated_simple} ({count_simple} removed)")

# Advanced mode
ablate_advanced = make_remove_expletives_with_coref(nlp)
ablated_advanced, count_advanced = ablate_advanced(doc)
print(f"Advanced: {ablated_advanced} ({count_advanced} removed)")
```

## Future Enhancements

Other ablations could similarly be extended:
- **remove_articles**: Context-aware article removal based on discourse coherence
- **lemmatize_verbs**: Preserve certain tenses in quoted speech
- **impoverish_determiners**: Preserve demonstratives in certain contexts

These would follow the same pattern: create a factory function that closes over additional configuration.
