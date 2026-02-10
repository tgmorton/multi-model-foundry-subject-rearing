"""
Clause structure annotator.

Extracts clause-level information including verb form, subject realization,
and object status for each clause in a sentence.
"""

from typing import Any, Dict, List, Optional

import spacy

from .base import BaseSentenceAnnotator

_FINITE_CLAUSE_DEPS = {"ROOT", "ccomp", "advcl", "acl:relcl", "acl", "xcomp"}


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

            dep = tok.dep_
            verb_forms = tok.morph.get("VerbForm")

            # Only process clause-heading verbs
            if dep not in _FINITE_CLAUSE_DEPS:
                continue

            is_finite = verb_forms and "Fin" in verb_forms
            is_infinitive = verb_forms and "Inf" in verb_forms

            # Skip if neither finite nor infinitive
            if not (is_finite or is_infinitive):
                continue

            # Analyze children for arguments
            children = {c.dep_: c for c in tok.children}
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

            # xcomp subject inheritance
            if subject_status == "none" and dep == "xcomp":
                head_children_deps = {c.dep_ for c in tok.head.children}
                if head_children_deps & {"nsubj", "nsubj:pass", "expl", "csubj", "csubj:pass"}:
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

        has_null_subject = any(
            c["subject_status"] == "none" and c["is_finite"]
            for c in clauses
        )

        return {"has_null_subject": has_null_subject}
