"""
Unit tests for preprocessing utility functions.

Tests device detection, token counting, checksumming, and environment capture.
"""

import pytest
from pathlib import Path
import hashlib

from preprocessing.utils import (
    count_tokens,
    compute_file_checksum,
    get_environment_info,
    ensure_directory_exists,
    count_files_in_directory
)


class TestCountTokens:
    """Tests for count_tokens function."""

    def test_count_simple_sentence(self):
        """Count tokens in a simple sentence."""
        text = "The cat sat on the mat."
        count = count_tokens(text)
        assert count == 6

    def test_count_with_punctuation(self):
        """Punctuation attached to words counts as one token."""
        text = "Hello, world!"
        count = count_tokens(text)
        assert count == 2  # "Hello," and "world!" are two tokens

    def test_count_empty_string(self):
        """Empty string has zero tokens."""
        assert count_tokens("") == 0

    def test_count_whitespace_only(self):
        """Whitespace-only string has zero tokens."""
        assert count_tokens("   \n  \t  ") == 0

    def test_count_multiple_lines(self):
        """Can count tokens across multiple lines."""
        text = "Line one.\nLine two.\nLine three."
        count = count_tokens(text)
        assert count == 6  # "Line" "one." "Line" "two." "Line" "three."

    def test_count_with_multiple_spaces(self):
        """Multiple spaces don't create extra tokens."""
        text = "word1    word2     word3"
        count = count_tokens(text)
        assert count == 3


class TestComputeFileChecksum:
    """Tests for compute_file_checksum function."""

    def test_compute_checksum_of_file(self, tmp_path):
        """Can compute SHA256 checksum of a file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, world!")

        checksum = compute_file_checksum(test_file)

        # Verify it's a valid hex string
        assert len(checksum) == 64  # SHA256 produces 64 hex characters
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_same_content_produces_same_checksum(self, tmp_path):
        """Same content produces same checksum."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        content = "This is test content."
        file1.write_text(content)
        file2.write_text(content)

        checksum1 = compute_file_checksum(file1)
        checksum2 = compute_file_checksum(file2)

        assert checksum1 == checksum2

    def test_different_content_produces_different_checksum(self, tmp_path):
        """Different content produces different checksum."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        file1.write_text("Content A")
        file2.write_text("Content B")

        checksum1 = compute_file_checksum(file1)
        checksum2 = compute_file_checksum(file2)

        assert checksum1 != checksum2

    def test_nonexistent_file_raises_error(self, tmp_path):
        """Computing checksum of nonexistent file raises FileNotFoundError."""
        nonexistent = tmp_path / "doesnotexist.txt"

        with pytest.raises(FileNotFoundError):
            compute_file_checksum(nonexistent)

    def test_can_use_different_algorithm(self, tmp_path):
        """Can use different hash algorithms."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, world!")

        checksum_md5 = compute_file_checksum(test_file, algorithm="md5")

        # MD5 produces 32 hex characters
        assert len(checksum_md5) == 32

    def test_invalid_algorithm_raises_error(self, tmp_path):
        """Invalid hash algorithm raises ValueError."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, world!")

        with pytest.raises(ValueError) as exc_info:
            compute_file_checksum(test_file, algorithm="invalid_algorithm")

        assert "Unsupported hash algorithm" in str(exc_info.value)

    def test_checksum_matches_manual_computation(self, tmp_path):
        """Checksum matches manual hashlib computation."""
        test_file = tmp_path / "test.txt"
        content = "Test content for checksum verification."
        test_file.write_text(content)

        # Compute using our function
        our_checksum = compute_file_checksum(test_file)

        # Compute manually
        manual_hash = hashlib.sha256()
        manual_hash.update(content.encode('utf-8'))
        manual_checksum = manual_hash.hexdigest()

        assert our_checksum == manual_checksum


class TestGetEnvironmentInfo:
    """Tests for get_environment_info function."""

    def test_returns_dict(self):
        """get_environment_info returns a dictionary."""
        info = get_environment_info()
        assert isinstance(info, dict)

    def test_includes_python_version(self):
        """Environment info includes Python version."""
        info = get_environment_info()
        assert "python_version" in info
        assert len(info["python_version"]) > 0

    def test_includes_platform(self):
        """Environment info includes platform."""
        info = get_environment_info()
        assert "platform" in info
        assert len(info["platform"]) > 0

    def test_includes_hostname(self):
        """Environment info includes hostname."""
        info = get_environment_info()
        assert "hostname" in info
        assert len(info["hostname"]) > 0


class TestEnsureDirectoryExists:
    """Tests for ensure_directory_exists function."""

    def test_creates_directory_if_not_exists(self, tmp_path):
        """Creates directory if it doesn't exist."""
        new_dir = tmp_path / "new_directory"
        assert not new_dir.exists()

        result = ensure_directory_exists(new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir.resolve()

    def test_does_not_error_if_exists(self, tmp_path):
        """Does not error if directory already exists."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()

        result = ensure_directory_exists(existing_dir)

        assert existing_dir.exists()
        assert result == existing_dir.resolve()

    def test_creates_nested_directories(self, tmp_path):
        """Creates nested directories (parents=True)."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        assert not nested_dir.exists()

        result = ensure_directory_exists(nested_dir)

        assert nested_dir.exists()
        assert nested_dir.is_dir()

    def test_returns_absolute_path(self, tmp_path):
        """Returns absolute path."""
        new_dir = tmp_path / "test_dir"
        result = ensure_directory_exists(new_dir)

        assert result.is_absolute()


class TestCountFilesInDirectory:
    """Tests for count_files_in_directory function."""

    def test_count_files_with_pattern(self, tmp_path):
        """Counts files matching pattern."""
        (tmp_path / "file1.train").touch()
        (tmp_path / "file2.train").touch()
        (tmp_path / "file3.txt").touch()

        count = count_files_in_directory(tmp_path, "*.train")

        assert count == 2

    def test_count_recursive(self, tmp_path):
        """Counts files recursively."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        (tmp_path / "file1.train").touch()
        (subdir / "file2.train").touch()

        count = count_files_in_directory(tmp_path, "*.train")

        assert count == 2

    def test_count_in_nonexistent_directory(self, tmp_path):
        """Returns 0 for nonexistent directory."""
        nonexistent = tmp_path / "doesnotexist"

        count = count_files_in_directory(nonexistent, "*.train")

        assert count == 0

    def test_count_with_no_matches(self, tmp_path):
        """Returns 0 when no files match pattern."""
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.md").touch()

        count = count_files_in_directory(tmp_path, "*.train")

        assert count == 0
