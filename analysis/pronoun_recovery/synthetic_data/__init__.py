"""Synthetic paired data generation for pronoun recovery training."""

from .data_format import SyntheticPair, load_pairs, save_pairs
from .generator import SyntheticPairGenerator

__all__ = ["SyntheticPairGenerator", "SyntheticPair", "load_pairs", "save_pairs"]
