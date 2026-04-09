# Developer Guide: Custom Ablations

Add your own linguistic ablations to the preprocessing pipeline.

## Overview

The preprocessing system uses a registry pattern
(`preprocessing/registry.py`). Adding a new ablation involves:

1. Implement an ablation callable (function or class).
2. Implement a validation function.
3. Register both with `AblationRegistry`.
4. Import the module from `preprocessing/ablations/__init__.py` so the
   registration runs at import time.
5. Write tests.

An ablation can be a **stateless function** or a **stateful class** — both
are supported by `AblationPipeline._process_file`
(`preprocessing/base.py:257`).

## Quick Start

Copy the template and modify:

```bash
cp preprocessing/ablations/template.py preprocessing/ablations/my_ablation.py
```

Add the import to `preprocessing/ablations/__init__.py`:

```python
from . import my_ablation  # triggers AblationRegistry.register
```

## Stateless Ablation Function

Signature: `(spacy.tokens.Doc) -> Tuple[str, int]`.

```python
from typing import Tuple
import spacy
from preprocessing.registry import AblationRegistry


def remove_adjectives_doc(doc: spacy.tokens.Doc) -> Tuple[str, int]:
    """Remove all adjectives (POS tag 'ADJ')."""
    parts = []
    num_removed = 0
    for token in doc:
        if token.pos_ == "ADJ":
            num_removed += 1
        else:
            parts.append(token.text_with_ws)
    return "".join(parts), num_removed


def validate_adjective_removal(original: str, ablated: str, nlp) -> bool:
    ablated_doc = nlp(ablated)
    return not any(t.pos_ == "ADJ" for t in ablated_doc)


AblationRegistry.register(
    "remove_adjectives",
    remove_adjectives_doc,
    validate_adjective_removal,
)
```

**Rules:**

- Use `token.text_with_ws` to preserve whitespace.
- Return `(text, num_items_modified)`.
- Do not mutate the `Doc` — it is immutable.
- Handle empty documents gracefully.

## Stateful Ablation Class

Stateful ablations (sentence-level removers that need line-index tracking,
context windows for coreference, per-file tier counts, lazy-loaded models)
should be implemented as a **callable class** with the same
`(Doc) -> (str, int)` call signature.

`AblationPipeline._process_file` recognises two optional hooks on the
registered callable:

| Hook                                      | Called when                       | Purpose                                                  |
|-------------------------------------------|-----------------------------------|----------------------------------------------------------|
| `reset_file_state() -> None`              | Start of every input file         | Clear per-file counters, context buffers, line indices   |
| `get_file_tier_counts() -> Dict[str,int]` | After the file has been processed | Return per-tier removal counts to be stored in the manifest |

If the callable exposes a `_removed_line_indices` attribute (a `List[int]`),
the pipeline also captures it into `FileStatistics.removed_line_indices`.

See `EnglishExpletiveSentenceRemover`
(`preprocessing/ablations/remove_expletive_sentences.py:57`) for a full
reference implementation.

### Minimal Stateful Template

```python
from typing import Dict, List, Tuple
import spacy
from preprocessing.registry import AblationRegistry


class MyStatefulAblation:
    def __init__(self) -> None:
        self._line_idx: int = 0
        self._removed_line_indices: List[int] = []
        self._tier_counts: Dict[str, int] = {"tier_a": 0, "tier_b": 0}

    # -- hooks picked up by AblationPipeline ------------------------------
    def reset_file_state(self) -> None:
        self._line_idx = 0
        self._removed_line_indices = []
        self._tier_counts = {k: 0 for k in self._tier_counts}

    def get_file_tier_counts(self) -> Dict[str, int]:
        return dict(self._tier_counts)

    # -- the ablation itself ----------------------------------------------
    def __call__(self, doc: spacy.tokens.Doc) -> Tuple[str, int]:
        idx = self._line_idx
        self._line_idx += 1

        tier = self._classify(doc)
        if tier is None:
            return doc.text_with_ws, 0

        self._tier_counts[tier] += 1
        self._removed_line_indices.append(idx)
        return "", 1

    def _classify(self, doc: spacy.tokens.Doc):
        # return "tier_a", "tier_b", or None
        ...


def validate_my_ablation(original: str, ablated: str, nlp) -> bool:
    return len(ablated) <= len(original)


AblationRegistry.register(
    "my_stateful_ablation",
    MyStatefulAblation(),
    validate_my_ablation,
)
```

**Important:** the instance is created at import time and reused across
files. `reset_file_state()` is what makes per-file reproducibility work —
do not rely on `__init__` running between files.

## Factory Pattern for Runtime Parameters

Use a factory when an ablation needs configuration (e.g. an optional model
name). The factory returns either a function or a class instance.

```python
def make_remove_expletive_sentences_en(
    coref_model: Optional[str] = None,
    context_lines: int = 3,
) -> EnglishExpletiveSentenceRemover:
    return EnglishExpletiveSentenceRemover(
        coref_model=coref_model,
        context_lines=context_lines,
    )
```

See `preprocessing/ablations/remove_expletive_sentences.py:304` for the
shipped example.

## Validation Function

```python
def validate(original: str, ablated: str, nlp) -> bool:
    """Return True if the ablation looks correct."""
    ...
```

Validators run in isolation (no access to stateful detector counters), so
for stateful ablations the validator typically re-runs a stateless version
of the detector. See `_make_validator`
(`preprocessing/ablations/remove_expletive_sentences.py:410`).

Validation errors are non-fatal — the pipeline logs a warning and
continues (`preprocessing/base.py:310`).

## Common Pitfalls

### Forgetting whitespace

```python
parts.append(token.text)           # WRONG — drops whitespace
parts.append(token.text_with_ws)   # RIGHT
parts.append("modified" + token.whitespace_)  # RIGHT for replacements
```

### Global module state

Use instance attributes on a class, not module-level globals. Module-level
state leaks across files and is not reset by `reset_file_state`.

### Assuming `__init__` runs per file

It does not. `AblationRegistry` stores a single instance, constructed once
at import time. Reset per-file state inside `reset_file_state()`.

### Mutating the Doc

spaCy Docs are immutable in this pipeline. Build new text by appending to
a list and joining.

## Testing

```python
# preprocessing/tests/test_my_ablation.py
import pytest
import spacy
from preprocessing.registry import AblationRegistry


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


def test_is_registered():
    assert AblationRegistry.is_registered("my_stateful_ablation")


def test_tier_counts_reset(nlp):
    ablate_fn, _ = AblationRegistry.get("my_stateful_ablation")
    ablate_fn.reset_file_state()
    ablate_fn(nlp("A trigger sentence."))
    counts_1 = ablate_fn.get_file_tier_counts()

    ablate_fn.reset_file_state()
    counts_2 = ablate_fn.get_file_tier_counts()
    assert all(v == 0 for v in counts_2.values())
```

Run:

```bash
python -m pytest preprocessing/tests/test_my_ablation.py -v
```

## Integration

Edit `preprocessing/ablations/__init__.py` so the module imports on
package load:

```python
from . import lemmatize_verbs
from . import remove_expletive_sentences
from . import impoverish_case
from . import enrich_verbal_morphology
from . import my_ablation  # <-- add this
```

After the import the ablation is usable via `AblationConfig`:

```python
config = AblationConfig(
    type="my_stateful_ablation",
    input_path="data/raw/",
    output_path="data/processed/",
    seed=42,
)
```

## Next Steps

- **Tier counting / provenance details**: see [Advanced](ADVANCED.md)
- **Testing patterns**: see [Testing Guide](TESTING.md)
- **Reference implementation**:
  `preprocessing/ablations/remove_expletive_sentences.py`
