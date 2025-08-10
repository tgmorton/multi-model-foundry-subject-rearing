"""
Language Model Evaluation Pipeline for Subject Drop Study.

This package provides comprehensive evaluation tools for assessing language models
trained on ablated corpora, focusing on the acquisition of the English overt 
subject constraint and related linguistic phenomena.

Main components:
- model_loader: Load and manage model checkpoints
- surprisal_calculator: Calculate surprisal for linguistic stimuli
- blimp_evaluator: Evaluate on BLIMP linguistic phenomena
- null_subject_evaluator: Evaluate on null-subject stimuli
- perplexity_evaluator: Calculate corpus perplexity
- evaluation_runner: Main orchestration script
- result_aggregator: Export results for statistical analysis

Example usage:
    python evaluation/evaluation_runner.py --config configs/evaluation_config.yaml
"""

__version__ = "1.0.0"