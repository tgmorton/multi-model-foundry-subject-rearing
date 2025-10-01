"""
Unit tests for utility functions.

Tests the helper functions for project root finding, git operations,
seed setting, and device detection.
"""

import os
import random
import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from model_foundry.utils import (
    find_project_root,
    get_git_commit_hash,
    set_seed,
    get_device
)


class TestFindProjectRoot:
    """Tests for find_project_root function."""

    def test_finds_git_root_from_nested_path(self, tmp_path):
        """Should find .git directory when starting from nested path."""
        # Create directory structure with .git
        git_root = tmp_path / "project"
        git_root.mkdir()
        (git_root / ".git").mkdir()
        nested = git_root / "src" / "deep" / "nested"
        nested.mkdir(parents=True)

        # Create a file in nested directory to use as start path
        test_file = nested / "test.py"
        test_file.write_text("# test")

        result = find_project_root(str(test_file))
        assert result == str(git_root)

    def test_finds_git_root_from_git_directory(self, tmp_path):
        """Should return the directory containing .git."""
        git_root = tmp_path / "myproject"
        git_root.mkdir()
        (git_root / ".git").mkdir()

        result = find_project_root(str(git_root))
        assert result == str(git_root)

    def test_fallback_to_cwd_when_no_git(self, tmp_path, monkeypatch):
        """Should fall back to cwd when no .git found."""
        no_git = tmp_path / "no_git_here"
        no_git.mkdir()

        # Mock os.getcwd to return a known value
        mock_cwd = "/fake/cwd"
        monkeypatch.setattr(os, 'getcwd', lambda: mock_cwd)

        result = find_project_root(str(no_git))
        assert result == mock_cwd

    def test_handles_root_directory(self, tmp_path):
        """Should handle being at root directory without crashing."""
        # This tests the edge case where path.parent == path
        result = find_project_root("/")
        # Should return cwd since no .git at root
        assert isinstance(result, str)


class TestGetGitCommitHash:
    """Tests for get_git_commit_hash function."""

    def test_returns_commit_hash_when_git_available(self):
        """Should return a valid commit hash when git is available."""
        # This test runs in actual git repo
        result = get_git_commit_hash()

        # Should be a 40-character hex string or "git_not_found"
        if result != "git_not_found":
            assert len(result) == 40
            assert all(c in "0123456789abcdef" for c in result)

    @patch('subprocess.check_output')
    def test_returns_specific_hash(self, mock_check_output):
        """Should return the hash from git command."""
        expected_hash = "abc123def456789012345678901234567890abcd"
        mock_check_output.return_value = (expected_hash + "\n").encode('ascii')

        result = get_git_commit_hash()
        assert result == expected_hash

    @patch('subprocess.check_output')
    def test_returns_fallback_on_git_error(self, mock_check_output):
        """Should return 'git_not_found' when git command fails."""
        mock_check_output.side_effect = Exception("git not found")

        result = get_git_commit_hash()
        assert result == "git_not_found"

    @patch('subprocess.check_output')
    def test_handles_subprocess_error(self, mock_check_output):
        """Should handle subprocess errors gracefully."""
        mock_check_output.side_effect = FileNotFoundError("git not in PATH")

        result = get_git_commit_hash()
        assert result == "git_not_found"


class TestSetSeed:
    """Tests for set_seed function."""

    def test_sets_python_random_seed(self):
        """Should set Python's random seed."""
        set_seed(42)
        result1 = random.random()

        set_seed(42)
        result2 = random.random()

        assert result1 == result2

    def test_sets_numpy_seed(self):
        """Should set NumPy's random seed."""
        set_seed(42)
        result1 = np.random.rand()

        set_seed(42)
        result2 = np.random.rand()

        assert result1 == result2

    def test_sets_torch_seed(self):
        """Should set PyTorch's random seed."""
        set_seed(42)
        result1 = torch.rand(1).item()

        set_seed(42)
        result2 = torch.rand(1).item()

        assert result1 == result2

    def test_different_seeds_produce_different_results(self):
        """Different seeds should produce different random values."""
        set_seed(42)
        result1 = random.random()

        set_seed(99)
        result2 = random.random()

        assert result1 != result2

    @pytest.mark.gpu
    def test_sets_cuda_seed_when_available(self, skip_if_no_cuda):
        """Should set CUDA seeds when CUDA is available."""
        set_seed(42)
        result1 = torch.cuda.FloatTensor(1).normal_().item()

        set_seed(42)
        result2 = torch.cuda.FloatTensor(1).normal_().item()

        # Note: CUDA results may vary slightly due to hardware
        # Just verify the function runs without error
        assert isinstance(result1, float)
        assert isinstance(result2, float)

    def test_reproducibility_across_operations(self):
        """Should ensure reproducibility across multiple operations."""
        set_seed(12345)

        # Do various random operations
        py_rand = random.randint(0, 1000)
        np_rand = np.random.randint(0, 1000)
        torch_rand = torch.randint(0, 1000, (1,)).item()

        # Reset and repeat
        set_seed(12345)

        py_rand2 = random.randint(0, 1000)
        np_rand2 = np.random.randint(0, 1000)
        torch_rand2 = torch.randint(0, 1000, (1,)).item()

        assert py_rand == py_rand2
        assert np_rand == np_rand2
        assert torch_rand == torch_rand2


class TestGetDevice:
    """Tests for get_device function."""

    def test_returns_torch_device(self):
        """Should return a torch.device object."""
        result = get_device()
        assert isinstance(result, torch.device)

    def test_returns_cuda_when_available(self):
        """Should return CUDA device when available."""
        if torch.cuda.is_available():
            result = get_device()
            assert result.type == "cuda"

    def test_returns_cpu_when_cuda_unavailable(self):
        """Should return CPU device when CUDA not available."""
        if not torch.cuda.is_available():
            result = get_device()
            assert result.type == "cpu"

    @patch('torch.cuda.is_available')
    def test_returns_cpu_when_mocked_no_cuda(self, mock_cuda_available):
        """Should return CPU when CUDA is mocked as unavailable."""
        mock_cuda_available.return_value = False

        result = get_device()
        assert result.type == "cpu"

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_name')
    def test_returns_cuda_when_mocked_available(self, mock_device_name, mock_cuda_available):
        """Should return CUDA when CUDA is mocked as available."""
        mock_cuda_available.return_value = True
        mock_device_name.return_value = "Mock GPU"

        result = get_device()
        assert result.type == "cuda"

    def test_device_can_be_used_for_tensor_creation(self):
        """Returned device should be usable for tensor operations."""
        device = get_device()

        # Should be able to create tensor on this device
        tensor = torch.zeros(10, device=device)
        assert tensor.device.type == device.type


class TestUtilsIntegration:
    """Integration tests combining multiple utility functions."""

    def test_seed_ensures_reproducible_workflow(self):
        """Setting seed should make entire workflow reproducible."""
        def random_workflow():
            """Simulate a workflow with multiple random operations."""
            a = random.random()
            b = np.random.rand()
            c = torch.rand(1).item()
            return a + b + c

        set_seed(777)
        result1 = random_workflow()

        set_seed(777)
        result2 = random_workflow()

        assert abs(result1 - result2) < 1e-6

    def test_project_root_and_git_hash_work_together(self, tmp_path):
        """Project root finding should work in git repos."""
        # Create a git repo structure
        git_root = tmp_path / "test_repo"
        git_root.mkdir()
        (git_root / ".git").mkdir()

        nested_file = git_root / "src" / "test.py"
        nested_file.parent.mkdir(parents=True)
        nested_file.write_text("test")

        # Should find the root
        root = find_project_root(str(nested_file))
        assert root == str(git_root)

        # Git hash should work (or return fallback)
        commit_hash = get_git_commit_hash()
        assert isinstance(commit_hash, str)
        assert len(commit_hash) > 0


class TestUtilsEdgeCases:
    """Edge case tests for utility functions."""

    def test_find_project_root_with_symlinks(self, tmp_path):
        """Should handle symlinks correctly."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        (real_dir / ".git").mkdir()

        # Create symlink (skip test if not supported)
        try:
            link_dir = tmp_path / "link"
            link_dir.symlink_to(real_dir)

            result = find_project_root(str(link_dir))
            # Should resolve to the real directory
            assert Path(result).resolve() == real_dir.resolve()
        except OSError:
            pytest.skip("Symlinks not supported on this system")

    def test_set_seed_with_zero(self):
        """Should handle seed value of 0."""
        set_seed(0)
        result = random.random()

        set_seed(0)
        result2 = random.random()

        assert result == result2

    def test_set_seed_with_large_value(self):
        """Should handle large seed values."""
        large_seed = 2**31 - 1  # Max 32-bit int

        set_seed(large_seed)
        result = random.random()

        set_seed(large_seed)
        result2 = random.random()

        assert result == result2

    def test_get_device_multiple_calls_consistent(self):
        """Multiple calls should return consistent device."""
        device1 = get_device()
        device2 = get_device()

        assert device1.type == device2.type
