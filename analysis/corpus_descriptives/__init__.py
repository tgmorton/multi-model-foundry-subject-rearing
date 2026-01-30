"""
Corpus Descriptive Analysis

Single-pass pipeline for Phase 1 descriptive analyses of English BabyLM corpus.
Processes each spaCy Doc through modular analyzers that accumulate counters
for pronoun inventory, expletives, verb finiteness, clause structure,
wh-questions, relative clauses, negation, and that-trace environments.
"""

__version__ = "0.1.0"
