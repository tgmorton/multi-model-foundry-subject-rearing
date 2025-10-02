import argparse
import os
from pathlib import Path

from .tokenizer_factory import train_tokenizer_from_config as factory_train


def find_project_root(start_path: str) -> str:
    """Finds the project root by searching upwards for a .git directory."""
    path = Path(start_path).resolve()
    while path.parent != path:
        if (path / '.git').is_dir():
            return str(path)
        path = path.parent
    print("Warning: .git directory not found. Falling back to current working directory as project root.")
    return os.getcwd()


def train_tokenizer_from_config(config_path: str):
    """
    Trains a tokenizer using parameters from a .yaml experiment file.

    This function now delegates to the tokenizer factory which supports
    multiple tokenizer types (SentencePiece, WordPiece, BPE, Character).

    Args:
        config_path: Path to the experiment's YAML configuration file
    """
    project_root = find_project_root(__file__)
    return factory_train(config_path, project_root)


def main():
    parser = argparse.ArgumentParser(
        description="Train a SentencePiece tokenizer for a specific experiment."
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the experiment's .yaml configuration file."
    )
    args = parser.parse_args()

    # Resolve the config path relative to the project root as well
    project_root = find_project_root(__file__)
    absolute_config_path = args.config_path if os.path.isabs(args.config_path) else os.path.join(project_root, args.config_path)

    if not os.path.exists(absolute_config_path):
        print(f"FATAL ERROR: Configuration file not found at '{absolute_config_path}'")
        return

    train_tokenizer_from_config(absolute_config_path)


if __name__ == '__main__':
    main()