"""Per-document pronoun insertion from model predictions.

Given a spaCy Doc (of the *ablated* text) and a list of per-token
predictions from the sequence labeler, this module inserts the
recovered pronouns immediately before each predicted verb, handling
sentence-initial capitalization adjustments.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import spacy

from ..constants import LABEL_NONE
from .gender_resolver import GenderResolver
from .pronoun_maps import get_default_pronoun, needs_gender_resolution

logger = logging.getLogger(__name__)


class PronounInserter:
    """Insert recovered pronouns into text based on model predictions.

    Instantiate once per language and reuse across documents.

    Args:
        language: ISO 639-1 language code (``"en"`` or ``"it"``).
    """

    def __init__(self, language: str = "en") -> None:
        self.language = language
        self.gender_resolver = GenderResolver(language=language)

    # ── Public API ───────────────────────────────────────────────────────

    def insert(
        self,
        doc: spacy.tokens.Doc,
        predictions: List[Dict[str, Any]],
    ) -> str:
        """Insert pronouns into a spaCy Doc based on model predictions.

        For each prediction the method:

        1. Looks up the lexical pronoun via :func:`.pronoun_maps.get_default_pronoun`
           (or calls :class:`.GenderResolver` for ``PRO.3sg``).
        2. Inserts the pronoun immediately before the verb token.
        3. Handles capitalization: if the verb was sentence-initial the
           pronoun is capitalized and the verb is lower-cased.
        4. Skips ``IMP`` / ``CONJ`` / ``NONE`` labels (no insertion).

        Args:
            doc: spaCy Doc of the (ablated) text.
            predictions: List of prediction dicts, each containing at minimum
                ``'token_idx'`` (int) and ``'label'`` (str).  Only non-NONE
                predictions need be included, but NONE entries are silently
                skipped if present.

        Returns:
            The text with recovered pronouns inserted.
        """
        # Build a lookup: token index -> resolved pronoun string
        insertion_map: Dict[int, str] = {}
        for pred in predictions:
            token_idx = pred["token_idx"]
            label = pred["label"]
            if token_idx < 0 or token_idx >= len(doc):
                logger.warning(
                    "Prediction token_idx %d out of range [0, %d); skipping",
                    token_idx,
                    len(doc),
                )
                continue
            verb_token = doc[token_idx]
            pronoun = self._resolve_pronoun(label, verb_token)
            if pronoun is not None:
                insertion_map[token_idx] = pronoun

        # Walk every token and build the output string
        parts: List[str] = []
        for token in doc:
            if token.i in insertion_map:
                pronoun = insertion_map[token.i]
                verb_text = token.text

                # Determine whether the verb is sentence-initial
                is_sent_start = token.is_sent_start or token.i == 0
                pronoun, verb_text = self._handle_capitalization(
                    pronoun, verb_text, is_sent_start
                )

                # Insert pronoun before the verb, separated by a space
                parts.append(pronoun)
                parts.append(" ")
                parts.append(verb_text)
            else:
                parts.append(token.text)

            # Preserve original trailing whitespace
            if token.whitespace_:
                parts.append(token.whitespace_)

        return "".join(parts)

    # ── Pronoun resolution ───────────────────────────────────────────────

    def _resolve_pronoun(
        self,
        label: str,
        verb_token: spacy.tokens.Token,
    ) -> Optional[str]:
        """Get the lexical pronoun to insert for a given label.

        Args:
            label: Predicted label string.
            verb_token: The spaCy token for the target verb.

        Returns:
            The pronoun string, or ``None`` if no insertion is needed
            (IMP, CONJ, NONE).
        """
        if label in ("IMP", "CONJ", LABEL_NONE):
            return None
        if needs_gender_resolution(label):
            return self.gender_resolver.resolve(verb_token)
        return get_default_pronoun(label, self.language)

    # ── Capitalization ───────────────────────────────────────────────────

    def _handle_capitalization(
        self,
        pronoun: str,
        verb_text: str,
        is_sent_start: bool,
    ) -> Tuple[str, str]:
        """Handle capitalization when inserting before a verb.

        If the verb was sentence-initial (first character uppercase, rest
        not all-caps) the pronoun inherits the capital and the verb is
        lower-cased.  This prevents output like ``*"go they"`` when the
        original was ``"Go ..."`` -- instead producing ``"They go ..."``.

        Args:
            pronoun: Lexical pronoun (lowercase).
            verb_text: Original verb surface form.
            is_sent_start: Whether the verb is at the start of a sentence.

        Returns:
            ``(formatted_pronoun, formatted_verb)`` tuple.
        """
        if is_sent_start and verb_text and verb_text[0].isupper():
            return pronoun.capitalize(), verb_text[0].lower() + verb_text[1:]
        return pronoun, verb_text
