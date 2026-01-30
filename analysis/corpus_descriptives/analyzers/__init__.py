"""
Corpus descriptive analyzers.

Each analyzer processes spaCy Docs and accumulates counters for one
aspect of the corpus descriptive analysis (Phase 1, §1.1–1.8).
"""

from .pronoun_inventory import PronounInventoryAnalyzer
from .expletives import ExpletiveAnalyzer
from .verb_finiteness import VerbFinitenessAnalyzer
from .clause_structure import ClauseStructureAnalyzer
from .wh_questions import WhQuestionAnalyzer
from .relative_clauses import RelativeClauseAnalyzer
from .negation import NegationAnalyzer
from .that_trace import ThatTraceAnalyzer

ANALYZER_REGISTRY = {
    "pronoun_inventory": PronounInventoryAnalyzer,
    "expletives": ExpletiveAnalyzer,
    "verb_finiteness": VerbFinitenessAnalyzer,
    "clause_structure": ClauseStructureAnalyzer,
    "wh_questions": WhQuestionAnalyzer,
    "relative_clauses": RelativeClauseAnalyzer,
    "negation": NegationAnalyzer,
    "that_trace": ThatTraceAnalyzer,
}

__all__ = [
    "ANALYZER_REGISTRY",
    "PronounInventoryAnalyzer",
    "ExpletiveAnalyzer",
    "VerbFinitenessAnalyzer",
    "ClauseStructureAnalyzer",
    "WhQuestionAnalyzer",
    "RelativeClauseAnalyzer",
    "NegationAnalyzer",
    "ThatTraceAnalyzer",
]
