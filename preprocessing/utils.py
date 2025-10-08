"""
Preprocessing Utilities

Shared utility functions for ablation pipelines, including device detection,
token counting, checksumming, environment info capture, and logging setup.

Consolidates common functionality from original preprocessing scripts.
Independent of model_foundry to avoid dependency conflicts.
"""

import hashlib
import logging
import os
import platform
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


def get_spacy_device(verbose: bool = False) -> str:
    """
    Detect and return the best available spaCy device.

    Checks for Apple Silicon (MPS), then CUDA, and defaults to CPU.
    This function is compatible with the device detection from the original
    preprocessing scripts.

    Args:
        verbose: If True, print detection information

    Returns:
        Device string: "mps", "cuda", or "cpu"
    """
    # Check environment variables
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    if verbose:
        print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")

    try:
        import torch
        if verbose:
            print(f"PyTorch version: {torch.__version__}")

        # Check for Apple Silicon (MPS)
        if torch.backends.mps.is_available():
            if verbose:
                print("Apple Silicon (MPS) device detected. Using GPU.")
            return "mps"

        # Check for CUDA
        if torch.cuda.is_available():
            if verbose:
                print("NVIDIA CUDA device detected. Using GPU.")
                print(f"CUDA device count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            return "cuda"
        else:
            if verbose:
                print("PyTorch CUDA not available")

    except ImportError as e:
        if verbose:
            print(f"PyTorch import error: {e}")
    except Exception as e:
        if verbose:
            print(f"GPU detection error: {e}")

    if verbose:
        print("Warning: No compatible GPU detected. spaCy will run on CPU, which may be slow.")
    return "cpu"


def count_tokens(text: str) -> int:
    """
    Count tokens using regex matching.

    This approach matches the token counting used in the original preprocessing
    scripts and aligns better with training tokenization than simple word splitting.

    Args:
        text: Text to tokenize and count

    Returns:
        Number of tokens found
    """
    # Use regex to split on whitespace, handling punctuation better
    tokens = re.findall(r'\S+', text)
    return len(tokens)


def compute_file_checksum(file_path: Path, algorithm: str = "sha256") -> str:
    """
    Compute checksum of a file.

    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use (default: sha256)

    Returns:
        Hexadecimal checksum string

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If algorithm is not supported
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get hash function
    try:
        hash_func = hashlib.new(algorithm)
    except ValueError:
        raise ValueError(
            f"Unsupported hash algorithm: {algorithm}. "
            f"Available: {', '.join(hashlib.algorithms_available)}"
        )

    # Compute checksum in chunks to handle large files
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def get_environment_info() -> dict:
    """
    Capture complete environment metadata for reproducibility.

    Returns:
        Dictionary with environment information including:
        - python_version
        - platform
        - hostname
        - spacy_version
        - pytorch_version (if available)
        - cuda_available (if PyTorch available)
    """
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
    }

    # Add spaCy version
    try:
        import spacy
        info["spacy_version"] = spacy.__version__
    except ImportError:
        info["spacy_version"] = "not installed"

    # Add PyTorch version if available
    try:
        import torch
        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        pass

    return info


def ensure_directory_exists(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Absolute path to the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def count_files_in_directory(directory: Path, pattern: str = "*.train") -> int:
    """
    Count files matching a pattern in a directory (recursively).

    Args:
        directory: Directory to search
        pattern: Glob pattern to match (default: "*.train")

    Returns:
        Number of matching files
    """
    directory = Path(directory)
    if not directory.exists():
        return 0

    return len(list(directory.rglob(pattern)))


def setup_logging(
    name: str,
    experiment: str = "default",
    phase: str = None,
    log_dir: Union[str, Path] = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Initialize logger with consistent format.

    Copied from model_foundry.logging_utils to avoid import dependency.

    Args:
        name: Logger name
        experiment: Experiment identifier
        phase: Optional phase name
        log_dir: Root directory for logs
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if logger.handlers:
        return logger

    log_dir = Path(log_dir) / experiment
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if phase:
        file_name = f"{experiment}_{phase}_{timestamp}.log"
    else:
        file_name = f"{experiment}_{timestamp}.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    file_handler = logging.FileHandler(log_dir / file_name)
    stream_handler = logging.StreamHandler(sys.stdout)

    for h in (file_handler, stream_handler):
        h.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False

    return logger


def find_project_root(start_path: str = __file__) -> Path:
    """
    Find project root by looking for marker files.

    Copied from model_foundry.utils to avoid import dependency.

    Args:
        start_path: Path to start searching from

    Returns:
        Path to project root

    Raises:
        FileNotFoundError: If no project root found
    """
    current = Path(start_path).resolve()

    if current.is_file():
        current = current.parent

    markers = ["pyproject.toml", "setup.py", ".git", "requirements.txt"]

    while current != current.parent:
        for marker in markers:
            if (current / marker).exists():
                return current
        current = current.parent

    # Fallback
    fallback = Path(start_path).resolve().parent.parent
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Could not find project root from {start_path}")
