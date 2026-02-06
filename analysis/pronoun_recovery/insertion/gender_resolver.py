"""Gender resolution heuristics for 3sg pronoun insertion.

Italian has overt morphological gender marking on past participles and
(in compound tenses) on the verb itself.  This module exposes a
:class:`GenderResolver` that inspects spaCy morphological features to
choose between ``lui`` / ``lei`` (Italian) or falls back to singular
``they`` (English).
"""

import logging
from typing import Optional

import spacy

logger = logging.getLogger(__name__)


class GenderResolver:
    """Resolve gender for 3sg pronouns using morphological and syntactic cues.

    Instantiate once per language and reuse across documents.

    Args:
        language: ISO 639-1 language code (``"en"`` or ``"it"``).
    """

    def __init__(self, language: str = "en") -> None:
        self.language = language

    # ── Public API ───────────────────────────────────────────────────────

    def resolve(self, verb_token: spacy.tokens.Token) -> str:
        """Determine the lexical pronoun for a 3sg null subject.

        Strategy by language:

        **Italian**

        1. Check past participle agreement: look for a ``Gender`` feature on
           children of the verb (e.g. *andata* -> Fem -> ``"lei"``).
        2. Check verb morphology: ``Gender`` feature on the verb itself
           (compound tenses where the participle is the syntactic head).
        3. Fallback: ``"lui"`` (masculine default).

        **English**

        1. (Future) Check for gendered coreferent in nearby context.
        2. Fallback: ``"they"`` (singular they).

        Args:
            verb_token: The spaCy token for the finite verb whose subject
                is to be recovered.

        Returns:
            The lexical pronoun string (e.g. ``"they"``, ``"lui"``,
            ``"lei"``).
        """
        if self.language == "it":
            return self._resolve_italian(verb_token)
        return self._resolve_english(verb_token)

    # ── English ──────────────────────────────────────────────────────────

    def _resolve_english(self, verb_token: spacy.tokens.Token) -> str:
        """English 3sg: default to singular 'they'.

        Future versions may inspect a coreference chain or nearby noun
        phrases, but for now the gender-neutral default is used
        unconditionally.
        """
        return "they"

    # ── Italian ──────────────────────────────────────────────────────────

    def _resolve_italian(self, verb_token: spacy.tokens.Token) -> str:
        """Italian 3sg: use morphological cues for gender resolution.

        The resolution strategy walks three levels:

        1. **Children of the verb** — Look for a past participle (or
           predicative adjective) that carries a ``Gender`` morph feature.
           Skip auxiliaries/copulas since they rarely carry gender
           agreement themselves.
        2. **Head of the verb** — If the verb itself is an auxiliary or
           copula, its head is the lexical verb / participle and may
           carry the gender feature.
        3. **Verb's own morphology** — In some compound tenses the
           gender feature propagates to the finite form.
        4. **Fallback** — ``"lui"`` (masculine default), consistent with
           the ``IT_DEFAULT_PRONOUN`` table in :mod:`..constants`.

        Args:
            verb_token: spaCy token for the finite verb.

        Returns:
            ``"lui"`` or ``"lei"``.
        """
        # 1. Check children for past participle with gender
        for child in verb_token.children:
            if child.pos_ == "VERB" and child.dep_ in ("aux", "aux:pass", "cop"):
                continue
            # Past participle as child (e.g., auxiliary + participle)
            gender_feats = child.morph.get("Gender")
            if gender_feats:
                gender = gender_feats[0]
                if gender == "Fem":
                    return "lei"
                elif gender == "Masc":
                    return "lui"

        # 2. Check head if verb is a participle under an auxiliary
        if verb_token.dep_ in ("aux", "cop"):
            head = verb_token.head
            gender_feats = head.morph.get("Gender")
            if gender_feats:
                gender = gender_feats[0]
                if gender == "Fem":
                    return "lei"
                elif gender == "Masc":
                    return "lui"

        # 3. Check verb's own gender feature
        verb_gender = verb_token.morph.get("Gender")
        if verb_gender:
            gender = verb_gender[0]
            if gender == "Fem":
                return "lei"
            elif gender == "Masc":
                return "lui"

        # 4. Fallback: masculine default
        return "lui"
