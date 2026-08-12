"""
Clause structure annotator.

Extracts clause-level information including verb form, subject realization,
and object status for each clause in a sentence.
"""

from typing import Any, Dict, List, Optional

import spacy

from preprocessing.dep_labels import normalize_dep

from .base import BaseSentenceAnnotator

_FINITE_CLAUSE_DEPS = {"ROOT", "ccomp", "advcl", "acl:relcl", "acl", "xcomp"}

_SERIAL_VERB_LEMMAS = frozenset({"go", "come", "run"})

_VOCATIVE_DISCOURSE_POS = {"INTJ", "PROPN"}
_VOCATIVE_DISCOURSE_DEPS = {"vocative", "discourse", "intj"}

_SUBJECT_DEPS = {"nsubj", "nsubj:pass", "expl", "csubj", "csubj:pass"}


def _is_nonroot_imperative(tok: spacy.tokens.Token, sent: spacy.tokens.Span) -> bool:
    """
    Detect non-ROOT verbs (advcl/ccomp) that are actually imperatives.

    Heuristic:
    - Must be VERB, or AUX with lemma "be"
    - No subject children, no "to" mark child
    - At sentence start (idx 0) or idx 1-2 if preceded only by
      INTJ/PROPN/vocative/discourse tokens
    """
    # Must be VERB, or AUX with lemma "be"
    if tok.pos_ == "AUX" and tok.lemma_.lower() == "be":
        pass
    elif tok.pos_ != "VERB":
        return False

    children_deps = {normalize_dep(c.dep_) for c in tok.children}

    # No overt subject
    if children_deps & _SUBJECT_DEPS:
        return False

    # No "to" infinitive mark
    if any(c.dep_ == "mark" and c.lemma_.lower() == "to" for c in tok.children):
        return False

    # Position heuristic: at or near sentence start
    rel_idx = tok.i - sent.start
    if rel_idx == 0:
        return True
    if rel_idx <= 2:
        return all(
            sent[j].pos_ in _VOCATIVE_DISCOURSE_POS
            or sent[j].dep_ in _VOCATIVE_DISCOURSE_DEPS
            for j in range(rel_idx)
        )
    return False


class ClauseStructureAnnotator(BaseSentenceAnnotator):
    """
    Annotates clause structure including subject/object realization.

    For each clause (verb with finite or infinitive form), extracts:
    - Clause type (ROOT, ccomp, advcl, xcomp, acl)
    - Verb information (lemma, finiteness)
    - Subject status (overt, none, expletive) and details
    - Direct object status and details
    - Indirect object status and details
    """

    output_fields = {
        "clauses": "List of clause annotation dicts",
    }

    def __init__(self, language: str = "en"):
        self.language = language

    def annotate_sentence(
        self,
        sent: spacy.tokens.Span,
        genre: str,
        speaker: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract clause structure annotations."""
        clauses = []

        for tok in sent:
            if tok.pos_ not in ("VERB", "AUX"):
                continue

            dep = normalize_dep(tok.dep_)
            verb_forms = tok.morph.get("VerbForm")

            # Only process clause-heading verbs
            if dep not in _FINITE_CLAUSE_DEPS:
                continue

            is_finite = verb_forms and "Fin" in verb_forms
            is_infinitive = verb_forms and "Inf" in verb_forms

            # Finiteness propagation: if the verb has a finite auxiliary
            # child (e.g., "don't know", "can't open", "doesn't want"),
            # the clause is functionally finite even if spaCy tags the
            # main verb as VerbForm=Inf.
            if not is_finite:
                for child in tok.children:
                    if normalize_dep(child.dep_) in ("aux", "aux:pass") and child.pos_ == "AUX":
                        child_vf = child.morph.get("VerbForm")
                        if child_vf and "Fin" in child_vf:
                            is_finite = True
                            is_infinitive = False
                            break

            # Skip if neither finite nor infinitive
            if not (is_finite or is_infinitive):
                continue

            # Analyze children for arguments (keys normalized to UD)
            children = {normalize_dep(c.dep_): c for c in tok.children}
            children_deps = set(children.keys())

            # Subject analysis
            subject_status = "none"
            subject_idx = None
            subject_lemma = None
            subject_person = None
            subject_number = None
            subject_is_pronoun = False

            if "expl" in children_deps:
                subject_status = "expletive"
                expl_tok = children.get("expl")
                if expl_tok:
                    subject_idx = expl_tok.i - sent.start
                    subject_lemma = expl_tok.lemma_.lower()
            elif children_deps & {"nsubj", "nsubj:pass"}:
                subject_status = "overt"
                nsubj_tok = children.get("nsubj") or children.get("nsubj:pass")
                if nsubj_tok:
                    subject_idx = nsubj_tok.i - sent.start
                    subject_lemma = nsubj_tok.lemma_.lower()
                    subject_is_pronoun = nsubj_tok.pos_ == "PRON"
                    person = nsubj_tok.morph.get("Person")
                    number = nsubj_tok.morph.get("Number")
                    subject_person = int(person[0]) if person else None
                    subject_number = number[0] if number else None
            elif children_deps & {"csubj", "csubj:pass"}:
                subject_status = "clausal"
                csubj_tok = children.get("csubj") or children.get("csubj:pass")
                if csubj_tok:
                    subject_idx = csubj_tok.i - sent.start
                    subject_lemma = csubj_tok.lemma_.lower()

            # xcomp subject inheritance — walk through ADJ intermediaries
            # and chain up through xcomp recursively (max 5 hops) to find
            # a head with an overt subject.
            if subject_status == "none" and dep == "xcomp":
                current_head = tok.head
                for _ in range(5):
                    # Skip through ADJ intermediaries (e.g., "able to swim",
                    # "hard to do") that sit between xcomp verb and the
                    # matrix clause with the subject.
                    if current_head.pos_ == "ADJ" and current_head.dep_ in (
                        "acomp", "attr", "ROOT", "ccomp", "advcl", "xcomp",
                    ):
                        current_head = current_head.head
                    head_children_deps = {normalize_dep(c.dep_) for c in current_head.children}
                    if head_children_deps & {"nsubj", "nsubj:pass", "expl", "csubj", "csubj:pass"}:
                        subject_status = "inherited"
                        break
                    # Continue chaining through xcomp (e.g., "want to try to get")
                    if current_head.dep_ == "xcomp":
                        current_head = current_head.head
                    else:
                        break

            # Non-ROOT imperative: advcl/ccomp at sentence start with
            # no subject — e.g., "look at this we got a problem"
            if subject_status == "none" and dep in ("advcl", "ccomp"):
                if self.language == "en":
                    if _is_nonroot_imperative(tok, sent):
                        subject_status = "imperative"
                else:
                    # For Italian, check Mood=Imp from morphology
                    mood = tok.morph.get("Mood")
                    if mood and "Imp" in mood:
                        subject_status = "imperative"

            # Serial verb subject sharing: "go get it", "come see this"
            # English-only — Italian uses prepositions (e.g., "vai a prendere")
            if subject_status == "none" and dep == "advcl" and self.language == "en":
                if tok.head.lemma_.lower() in _SERIAL_VERB_LEMMAS:
                    has_to = any(
                        c.dep_ == "mark" and c.lemma_.lower() == "to"
                        for c in tok.children
                    )
                    if not has_to and tok.i == tok.head.i + 1:
                        subject_status = "inherited"

            # Disfluent verb reduplication
            if subject_status == "none" and tok.i > sent.start:
                prev = tok.doc[tok.i - 1]
                if prev.lemma_.lower() == tok.lemma_.lower() and prev.pos_ in ("VERB", "AUX"):
                    subject_status = "disfluent_copy"

            # Direct object analysis
            object_status = "none"
            object_idx = None
            object_lemma = None
            object_is_pronoun = False

            if "obj" in children_deps:
                object_status = "overt"
                obj_tok = children.get("obj")
                if obj_tok:
                    object_idx = obj_tok.i - sent.start
                    object_lemma = obj_tok.lemma_.lower()
                    object_is_pronoun = obj_tok.pos_ == "PRON"
            elif "ccomp" in children_deps:
                object_status = "clause"

            # Indirect object analysis
            iobject_status = "none"
            iobject_idx = None
            iobject_lemma = None
            iobject_is_pronoun = False

            if "iobj" in children_deps:
                iobject_status = "overt"
                iobj_tok = children.get("iobj")
                if iobj_tok:
                    iobject_idx = iobj_tok.i - sent.start
                    iobject_lemma = iobj_tok.lemma_.lower()
                    iobject_is_pronoun = iobj_tok.pos_ == "PRON"

            clause = {
                "clause_type": dep,
                "verb_idx": tok.i - sent.start,
                "verb_lemma": tok.lemma_.lower(),
                "is_finite": is_finite,
                # Subject
                "subject_status": subject_status,
                "subject_idx": subject_idx,
                "subject_lemma": subject_lemma,
                "subject_person": subject_person,
                "subject_number": subject_number,
                "subject_is_pronoun": subject_is_pronoun,
                # Direct object
                "object_status": object_status,
                "object_idx": object_idx,
                "object_lemma": object_lemma,
                "object_is_pronoun": object_is_pronoun,
                # Indirect object
                "iobject_status": iobject_status,
                "iobject_idx": iobject_idx,
                "iobject_lemma": iobject_lemma,
                "iobject_is_pronoun": iobject_is_pronoun,
            }
            clauses.append(clause)

        return {"clauses": clauses}

    def get_sentence_flags(
        self,
        annotations: Dict[str, Any],
    ) -> Dict[str, bool]:
        """Compute clause-related flags."""
        clauses = annotations.get("clauses", [])

        _null_statuses = {"none", "inherited", "imperative"}
        has_null_subject_finite = any(
            c["subject_status"] in _null_statuses and c["is_finite"]
            for c in clauses
        )
        has_null_subject_nonfinite = any(
            c["subject_status"] in _null_statuses and not c["is_finite"]
            for c in clauses
        )

        return {
            "has_null_subject": has_null_subject_finite or has_null_subject_nonfinite,
            "has_null_subject_finite": has_null_subject_finite,
            "has_null_subject_nonfinite": has_null_subject_nonfinite,
        }
