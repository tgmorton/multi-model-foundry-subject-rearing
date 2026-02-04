"""
Complexity annotator.

Measures sentence complexity including clause count, embedding depth,
and coordination.
"""

from typing import Any, Dict, List, Optional, Set

import spacy

from .base import BaseSentenceAnnotator


class ComplexityAnnotator(BaseSentenceAnnotator):
    """
    Annotates sentence complexity metrics.

    Extracts:
    - Number of clauses (finite + non-finite)
    - Maximum embedding depth
    - Coordination presence and count
    - Fragment/imperative status
    """

    output_fields = {
        "complexity": "Complexity metrics dict",
    }

    def annotate_sentence(
        self,
        sent: spacy.tokens.Span,
        genre: str,
        speaker: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract complexity annotations."""
        # Count clauses (verbs that head clauses)
        clause_deps = {"ROOT", "ccomp", "xcomp", "advcl", "acl", "acl:relcl", "parataxis"}
        n_clauses = 0
        clause_verbs = []

        for tok in sent:
            if tok.pos_ in ("VERB", "AUX") and tok.dep_ in clause_deps:
                n_clauses += 1
                clause_verbs.append(tok)

        # Handle edge case: no clauses found (fragment)
        if n_clauses == 0:
            n_clauses = 1  # Count the sentence itself as a clause

        # Calculate max embedding depth using dependency tree
        max_depth = self._calculate_max_depth(sent)

        # Detect coordination
        has_coordination = False
        n_conjuncts = 0

        for tok in sent:
            if tok.dep_ == "conj":
                has_coordination = True
                n_conjuncts += 1

        # Detect fragments and imperatives
        root = None
        for tok in sent:
            if tok.dep_ == "ROOT":
                root = tok
                break

        is_fragment = False
        is_imperative = False

        if root is None:
            is_fragment = True
        elif root.pos_ not in ("VERB", "AUX"):
            is_fragment = True
        else:
            # Check for finite verb
            verb_forms = root.morph.get("VerbForm")
            mood = root.morph.get("Mood")

            if verb_forms and "Fin" not in verb_forms:
                # Non-finite root - could be fragment
                is_fragment = True
            elif mood and "Imp" in mood:
                is_imperative = True
            elif verb_forms and "Fin" in verb_forms:
                # Finite root without subject could be imperative
                has_subject = any(
                    c.dep_ in ("nsubj", "nsubj:pass", "expl")
                    for c in root.children
                )
                if not has_subject:
                    is_imperative = True

        complexity = {
            "n_clauses": n_clauses,
            "max_embedding_depth": max_depth,
            "has_coordination": has_coordination,
            "n_conjuncts": n_conjuncts,
        }

        return {
            "complexity": complexity,
            "is_fragment": is_fragment,
            "is_imperative": is_imperative,
        }

    def _calculate_max_depth(self, sent: spacy.tokens.Span) -> int:
        """Calculate maximum embedding depth in dependency tree."""
        # Find the root
        root = None
        for tok in sent:
            if tok.dep_ == "ROOT":
                root = tok
                break

        if root is None:
            return 1

        # BFS to find max depth
        max_depth = 0
        stack = [(root, 0)]
        visited: Set[int] = set()

        while stack:
            tok, depth = stack.pop()
            if tok.i in visited:
                continue
            visited.add(tok.i)

            max_depth = max(max_depth, depth)

            for child in tok.children:
                if child.i >= sent.start and child.i < sent.end:
                    stack.append((child, depth + 1))

        return max_depth

    def get_sentence_flags(
        self,
        annotations: Dict[str, Any],
    ) -> Dict[str, bool]:
        """Return fragment and imperative flags."""
        return {
            "is_fragment": annotations.get("is_fragment", False),
            "is_imperative": annotations.get("is_imperative", False),
        }
