"""
Topic continuity annotator.

Extracts information about the main clause subject for topic continuity analysis.
"""

from typing import Any, Dict, List, Optional

import spacy

from preprocessing.dep_labels import normalize_dep

from .base import BaseSentenceAnnotator


class TopicAnnotator(BaseSentenceAnnotator):
    """
    Annotates topic continuity information.

    Extracts the root clause subject for tracking discourse topic across sentences.
    This enables analysis of:
    - Pronoun use in topic-continuing contexts
    - Null subject distribution by topic status
    - Topic shift patterns
    """

    output_fields = {
        "topic_info": "Topic continuity information dict",
    }

    def annotate_sentence(
        self,
        sent: spacy.tokens.Span,
        genre: str,
        speaker: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract topic continuity annotations."""
        # Find the root verb
        root_verb = None
        for tok in sent:
            if tok.dep_ == "ROOT":
                root_verb = tok
                break

        topic_info = {
            "root_subject_lemma": None,
            "root_subject_is_pronoun": False,
            "root_subject_person": None,
        }

        if root_verb is None:
            return {"topic_info": topic_info}

        # Find subject of root verb
        for child in root_verb.children:
            if normalize_dep(child.dep_) in ("nsubj", "nsubj:pass"):
                topic_info["root_subject_lemma"] = child.lemma_.lower()
                topic_info["root_subject_is_pronoun"] = child.pos_ == "PRON"

                person = child.morph.get("Person")
                if person:
                    topic_info["root_subject_person"] = int(person[0])
                break
            elif child.dep_ == "expl":
                # Expletive subject - not a true topic
                topic_info["root_subject_lemma"] = child.lemma_.lower()
                topic_info["root_subject_is_pronoun"] = True
                break

        return {"topic_info": topic_info}

    def get_sentence_flags(
        self,
        annotations: Dict[str, Any],
    ) -> Dict[str, bool]:
        """No flags for topic annotator."""
        return {}
