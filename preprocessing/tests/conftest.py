"""
Pytest configuration and shared fixtures for preprocessing tests.

Provides common fixtures for testing ablation pipelines, including
sample text data, temporary directories, and mock ablation functions.
"""

import pytest
from pathlib import Path
from typing import Tuple

from preprocessing.registry import AblationRegistry


@pytest.fixture(autouse=True)
def reset_registry():
    """Automatically clear registry before and after each test."""
    AblationRegistry.clear()
    yield
    AblationRegistry.clear()


@pytest.fixture
def sample_corpus_dir(tmp_path):
    """
    Create a sample corpus directory with test files.

    Returns:
        Path to temporary directory containing:
        - file1.train: 3 lines of text
        - file2.train: 2 lines of text
        - subdir/file3.train: 1 line of text
    """
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    # Create sample files
    file1 = corpus_dir / "file1.train"
    file1.write_text("The cat sat on the mat.\nThe dog ran in the park.\nBirds fly in the sky.\n")

    file2 = corpus_dir / "file2.train"
    file2.write_text("Hello world.\nGoodbye world.\n")

    # Create subdirectory with file
    subdir = corpus_dir / "subdir"
    subdir.mkdir()
    file3 = subdir / "file3.train"
    file3.write_text("A single line of text.\n")

    return corpus_dir


@pytest.fixture
def sample_pool_dir(tmp_path):
    """
    Create a sample replacement pool directory.

    Returns:
        Path to temporary directory with replacement pool files
    """
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()

    # Create corresponding pool files
    file1 = pool_dir / "file1.train"
    file1.write_text("Pool sentence one.\nPool sentence two.\nPool sentence three.\n")

    file2 = pool_dir / "file2.train"
    file2.write_text("Extra pool content.\n")

    subdir = pool_dir / "subdir"
    subdir.mkdir()
    file3 = subdir / "file3.train"
    file3.write_text("More pool data.\n")

    return pool_dir


@pytest.fixture
def dummy_ablation_function():
    """
    Create a simple ablation function for testing.

    This function removes all instances of "the" (case-insensitive).
    """
    def ablate(doc) -> Tuple[str, int]:
        """Remove all instances of 'the' from text."""
        text = doc.text_with_ws if hasattr(doc, 'text_with_ws') else str(doc)
        original_text = text
        modified_text = ""
        num_removed = 0

        # Simple word-by-word processing
        words = text.split()
        for word in words:
            if word.lower().strip(".,!?") == "the":
                num_removed += 1
            else:
                modified_text += word + " "

        return modified_text.rstrip() + "\n" if modified_text else "", num_removed

    return ablate


@pytest.fixture
def dummy_validator_function():
    """
    Create a simple validator function for testing.

    Validates that the ablated text is shorter than the original.
    """
    def validate(original: str, ablated: str, nlp) -> bool:
        """Check that ablated text is shorter."""
        return len(ablated) <= len(original)

    return validate


@pytest.fixture
def mock_spacy_doc():
    """
    Create a mock spaCy Doc object for testing.

    This mock has basic attributes needed for ablation functions.
    """
    class MockToken:
        def __init__(self, text, whitespace=" "):
            self.text = text
            self.text_with_ws = text + whitespace
            self.lower_ = text.lower()
            self.pos_ = "NOUN"  # Default POS tag
            self.dep_ = "ROOT"  # Default dependency

    class MockDoc:
        def __init__(self, text):
            self.text = text
            self.text_with_ws = text
            # Simple tokenization
            words = text.split()
            self.tokens = [MockToken(word) for word in words]

        def __iter__(self):
            return iter(self.tokens)

    return MockDoc


@pytest.fixture
def empty_corpus_dir(tmp_path):
    """
    Create an empty corpus directory.

    Returns:
        Path to empty temporary directory
    """
    corpus_dir = tmp_path / "empty_corpus"
    corpus_dir.mkdir()
    return corpus_dir


@pytest.fixture
def single_file_corpus(tmp_path):
    """
    Create a corpus with a single file.

    Returns:
        Path to temporary directory with one .train file
    """
    corpus_dir = tmp_path / "single_corpus"
    corpus_dir.mkdir()

    file1 = corpus_dir / "single.train"
    file1.write_text("This is a test sentence.\nAnother test sentence.\n")

    return corpus_dir
