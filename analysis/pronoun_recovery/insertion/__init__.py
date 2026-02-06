"""Stage 2: Pronoun insertion from model predictions (token classifier and seq2seq)."""

from .corpus_processor import CorpusInsertionPipeline
from .gender_resolver import GenderResolver
from .inserter import PronounInserter
from .pronoun_maps import get_default_pronoun, needs_gender_resolution
from .seq2seq_processor import Seq2SeqInsertionPipeline

__all__ = [
    "CorpusInsertionPipeline",
    "GenderResolver",
    "PronounInserter",
    "Seq2SeqInsertionPipeline",
    "get_default_pronoun",
    "needs_gender_resolution",
]
