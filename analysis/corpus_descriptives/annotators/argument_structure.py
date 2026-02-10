"""
Argument structure annotator.

Provides comprehensive tracking of all core arguments (subject, direct object,
indirect object) for each verb, including transitivity classification.
"""

from typing import Any, Dict, List, Optional

import spacy

from .base import BaseSentenceAnnotator


class ArgumentStructureAnnotator(BaseSentenceAnnotator):
    """
    Annotates argument structure for each verb in a sentence.

    For each verb, extracts:
    - Transitivity classification (intransitive, transitive, ditransitive)
    - Subject information (status, pronominality, person, number)
    - Direct object information
    - Indirect object information

    This is more comprehensive than ClauseStructureAnnotator as it tracks
    all verbs (not just clause-heading ones) and provides detailed
    argument properties.
    """

    output_fields = {
        "argument_structure": "List of argument structure annotation dicts",
    }

    def annotate_sentence(
        self,
        sent: spacy.tokens.Span,
        genre: str,
        speaker: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract argument structure annotations."""
        argument_structures = []

        for tok in sent:
            if tok.pos_ not in ("VERB", "AUX"):
                continue

            # Skip non-head verbs (e.g., auxiliaries in verb chains)
            # unless they are the main predicate
            if tok.dep_ in ("aux", "auxpass") and tok.head.pos_ in ("VERB", "AUX"):
                continue

            children = {c.dep_: c for c in tok.children}
            children_deps = set(children.keys())

            # === Subject analysis ===
            subject_status = "null"
            subject_idx = None
            subject_lemma = None
            subject_is_pronoun = False
            subject_person = None
            subject_number = None

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
            if subject_status == "null" and tok.dep_ == "xcomp":
                head_children_deps = {c.dep_ for c in tok.head.children}
                if head_children_deps & {"nsubj", "nsubj:pass", "expl", "csubj", "csubj:pass"}:
                    subject_status = "inherited"

            # Disfluent verb reduplication
            if subject_status == "null" and tok.i > sent.start:
                prev = tok.doc[tok.i - 1]
                if prev.lemma_.lower() == tok.lemma_.lower() and prev.pos_ in ("VERB", "AUX"):
                    subject_status = "disfluent_copy"

            # === Direct object analysis ===
            object_status = "null"
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
            elif "ccomp" in children_deps or "xcomp" in children_deps:
                object_status = "clausal"

            # === Indirect object analysis ===
            iobject_status = "null"
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

            # === Transitivity classification ===
            has_obj = object_status in ("overt", "clausal")
            has_iobj = iobject_status == "overt"

            if has_iobj:
                transitivity = "ditransitive"
            elif has_obj:
                transitivity = "transitive"
            else:
                transitivity = "intransitive"

            arg_struct = {
                "verb_idx": tok.i - sent.start,
                "verb_lemma": tok.lemma_.lower(),
                "transitivity": transitivity,
                # Subject
                "subject_status": subject_status,
                "subject_idx": subject_idx,
                "subject_lemma": subject_lemma,
                "subject_is_pronoun": subject_is_pronoun,
                "subject_person": subject_person,
                "subject_number": subject_number,
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
            argument_structures.append(arg_struct)

        return {"argument_structure": argument_structures}

    def get_sentence_flags(
        self,
        annotations: Dict[str, Any],
    ) -> Dict[str, bool]:
        """Compute argument structure flags."""
        structures = annotations.get("argument_structure", [])

        has_null_subject = any(s["subject_status"] == "null" for s in structures)
        has_null_object = any(
            s["object_status"] == "null" and s["transitivity"] != "intransitive"
            for s in structures
        )
        has_ditransitive = any(s["transitivity"] == "ditransitive" for s in structures)

        return {
            "has_null_subject": has_null_subject,
            "has_null_object": has_null_object,
            "has_ditransitive": has_ditransitive,
        }
