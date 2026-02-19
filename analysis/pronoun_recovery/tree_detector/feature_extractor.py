"""
Per-verb feature extraction for Italian null subject detection.

Takes a spaCy Doc, runs existing annotators, and produces a flat feature
dict per finite verb suitable for decision tree training.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import spacy.tokens

from analysis.corpus_descriptives.annotators import (
    CompositeAnnotator,
    get_default_annotators,
)
from analysis.corpus_descriptives.constants import (
    IMPERSONAL_VERBS_IT,
    NECESSITY_VERBS_IT,
    WEATHER_VERBS_IT,
)
from analysis.pronoun_recovery.constants import MORPH_TO_LABEL_SUFFIX


# Italian modal verb lemmas
_MODAL_LEMMAS_IT = frozenset({
    "potere", "dovere", "volere", "sapere",
})

# Italian essere/avere
_ESSERE = "essere"
_AVERE = "avere"


@dataclass
class VerbFeatureRow:
    """A finite verb with its extracted features and metadata."""

    token_idx: int
    char_offset: int
    verb_text: str
    verb_lemma: str
    person: Optional[str]
    number: Optional[str]
    morph_label_suffix: Optional[str]
    features: Dict[str, Any] = field(default_factory=dict)


class ItalianVerbFeatureExtractor:
    """Extract feature vectors from Italian sentences for tree training.

    Runs the full annotator suite once per sentence, then joins annotator
    outputs to each finite verb position to build a flat feature dict.
    """

    def __init__(self, language: str = "it"):
        self.language = language
        self._annotators = get_default_annotators(language=language)
        self._composite = CompositeAnnotator(self._annotators)

    def extract_sentence(
        self,
        sent: spacy.tokens.Span,
        genre: str = "unknown",
    ) -> List[VerbFeatureRow]:
        """Extract feature rows for all finite verbs in a sentence.

        Args:
            sent: spaCy Span (a sentence).
            genre: Corpus genre label for the annotators.

        Returns:
            List of VerbFeatureRow, one per finite verb found.
        """
        # Run all annotators once
        annotations = self._composite.annotate_sentence(sent, genre=genre)

        # Index annotator outputs by token position for fast lookup
        verb_annots = self._index_by_token_idx(annotations.get("verbs", []))
        clause_annots = self._index_by_verb_idx(annotations.get("clauses", []))
        argstruct_annots = self._index_by_verb_idx(
            annotations.get("argument_structure", [])
        )

        # Sentence-level features (replicated across all verbs)
        complexity = annotations.get("complexity", {})
        is_fragment = annotations.get("is_fragment", False)
        is_imperative = annotations.get("is_imperative", False)

        # Collect per-token indices for pronouns, expletives, negations, etc.
        pronouns = annotations.get("pronouns", [])
        expletives = annotations.get("expletives", [])
        negations = annotations.get("negations", [])
        wh_extractions = annotations.get("wh_extractions", [])
        relative_clauses = annotations.get("relative_clauses", [])
        bridge_complements = annotations.get("bridge_complements", [])
        topic_info = annotations.get("topic_info", {})

        rows: List[VerbFeatureRow] = []

        for tok in sent:
            if tok.pos_ not in ("VERB", "AUX"):
                continue

            verb_forms = tok.morph.get("VerbForm")
            if not verb_forms or verb_forms[0] != "Fin":
                continue

            # Morphology
            person_feats = tok.morph.get("Person")
            number_feats = tok.morph.get("Number")
            person = person_feats[0] if person_feats else None
            number = number_feats[0] if number_feats else None
            morph_suffix = (
                MORPH_TO_LABEL_SUFFIX.get((person, number))
                if person and number
                else None
            )

            # Build flat feature dict
            feats: Dict[str, Any] = {}

            # --- Verb morphology ---
            tense_vals = tok.morph.get("Tense")
            mood_vals = tok.morph.get("Mood")
            aspect_vals = tok.morph.get("Aspect")

            feats["verb_tense"] = tense_vals[0] if tense_vals else "unknown"
            feats["verb_mood"] = mood_vals[0] if mood_vals else "unknown"
            feats["verb_aspect"] = aspect_vals[0] if aspect_vals else "unknown"
            feats["verb_person"] = person or "unknown"
            feats["verb_number"] = number or "unknown"

            # From VerbAnnotator (if matched)
            rel_idx = tok.i - sent.start
            v_annot = verb_annots.get(rel_idx, {})
            feats["verb_clause_position"] = v_annot.get("clause_position", "other")

            # --- Clause structure ---
            c_annot = clause_annots.get(rel_idx, {})
            feats["clause_type"] = c_annot.get("clause_type", tok.dep_)
            feats["dep_label"] = tok.dep_
            feats["clause_subject_status"] = c_annot.get("subject_status", "unknown")
            feats["clause_is_finite"] = c_annot.get("is_finite", True)

            # --- Argument structure ---
            a_annot = argstruct_annots.get(rel_idx, {})
            feats["transitivity"] = a_annot.get("transitivity", "unknown")
            feats["object_status"] = a_annot.get("object_status", "unknown")
            feats["iobject_status"] = a_annot.get("iobject_status", "unknown")
            feats["argstruct_subject_status"] = a_annot.get(
                "subject_status", "unknown"
            )

            # --- Clitics / Pronouns ---
            verb_pronouns = [
                p for p in pronouns if p.get("head_verb_idx") == rel_idx
            ]
            feats["has_clitic_on_verb"] = any(
                p.get("is_clitic", False) for p in verb_pronouns
            )
            feats["has_reflexive_clitic"] = any(
                p.get("is_reflexive", False) for p in verb_pronouns
            )
            feats["n_clitics"] = sum(
                1 for p in verb_pronouns if p.get("is_clitic", False)
            )

            # --- Expletive context ---
            lemma = tok.lemma_.lower()
            feats["is_weather_verb"] = lemma in WEATHER_VERBS_IT
            feats["is_impersonal_verb"] = lemma in IMPERSONAL_VERBS_IT
            feats["is_necessity_verb"] = lemma in NECESSITY_VERBS_IT

            # --- Copular impersonal ---
            # "è importante/necessario/chiaro..." — 3p cop + ADJ/NOUN head
            # + no nsubj on the head → impersonal, NOT referential null subject
            feats["is_copular_impersonal"] = self._is_copular_impersonal(
                tok, person
            )

            # --- Sentence structure (complexity) ---
            feats["n_clauses"] = complexity.get("n_clauses", 1)
            feats["max_embedding_depth"] = complexity.get("max_embedding_depth", 0)
            feats["has_coordination"] = complexity.get("has_coordination", False)

            # --- Fragment / imperative ---
            feats["is_fragment"] = is_fragment
            feats["is_imperative"] = is_imperative

            # --- Wh-context ---
            # Check if this verb is in a wh-clause
            feats["is_in_wh_clause"] = any(
                wh.get("clause_level") == "root"
                and self._verb_is_head_of_wh(tok, wh, sent)
                for wh in wh_extractions
            )
            feats["wh_extraction_type"] = self._get_wh_extraction_type(
                tok, wh_extractions, sent
            )

            # --- Relative clause ---
            feats["is_in_relcl"] = tok.dep_ in ("acl:relcl", "acl")
            rel_match = next(
                (rc for rc in relative_clauses if rc.get("verb_idx") == rel_idx),
                None,
            )
            feats["rel_type"] = rel_match["rel_type"] if rel_match else "none"

            # Relative pronoun as subject: in subject-extracted relative
            # clauses ("la commissione che decide..."), "che" IS the subject.
            # The verb is NOT a null-subject position.
            feats["rel_pronoun_is_subject"] = (
                feats["is_in_relcl"]
                and rel_match is not None
                and rel_match.get("rel_type") == "subject"
            )

            # --- Bridge complement ---
            bridge_match = next(
                (
                    bc
                    for bc in bridge_complements
                    if bc.get("matrix_verb_idx") == rel_idx
                ),
                None,
            )
            feats["is_bridge_complement"] = tok.dep_ == "ccomp" and any(
                bc.get("matrix_verb_idx") is not None for bc in bridge_complements
            )
            feats["has_complementizer"] = (
                bridge_match["comp_present"] if bridge_match else False
            )

            # --- Negation ---
            verb_negations = [
                n
                for n in negations
                if n.get("subject_status") != "n/a"
                and self._negation_heads_verb(n, tok, sent)
            ]
            feats["has_negation"] = len(verb_negations) > 0
            feats["negation_position"] = (
                verb_negations[0]["position"] if verb_negations else "none"
            )

            # --- Discourse / Topic ---
            feats["prev_root_subject_person"] = topic_info.get(
                "root_subject_person"
            )
            feats["topic_matches_verb_morph"] = (
                topic_info.get("root_subject_person") is not None
                and person is not None
                and str(topic_info.get("root_subject_person")) == str(person)
            )

            # --- Positional ---
            sent_len = len(sent)
            feats["verb_position_normalized"] = (
                rel_idx / sent_len if sent_len > 0 else 0.0
            )
            feats["is_sentence_initial_verb"] = rel_idx == 0
            feats["distance_to_root"] = self._distance_to_root(tok)

            # --- Verb identity ---
            feats["is_essere"] = lemma == _ESSERE
            feats["is_avere"] = lemma == _AVERE
            feats["is_modal"] = lemma in _MODAL_LEMMAS_IT

            # --- Subject presence (from dependency tree directly) ---
            children_deps = {c.dep_ for c in tok.children}
            feats["has_overt_nsubj"] = bool(
                children_deps & {"nsubj", "nsubj:pass"}
            )
            feats["has_expl"] = "expl" in children_deps
            feats["has_csubj"] = bool(
                children_deps & {"csubj", "csubj:pass"}
            )
            feats["is_xcomp"] = tok.dep_ == "xcomp"

            # --- Aux/cop: check head verb's subject ---
            is_aux_or_cop = tok.dep_ in ("aux", "aux:pass", "cop")
            feats["is_aux_or_cop"] = is_aux_or_cop
            if is_aux_or_cop and tok.head is not None:
                head_children_deps = {c.dep_ for c in tok.head.children}
                feats["head_has_overt_nsubj"] = bool(
                    head_children_deps & {"nsubj", "nsubj:pass"}
                )
                feats["head_has_expl"] = "expl" in head_children_deps
                feats["head_has_csubj"] = bool(
                    head_children_deps & {"csubj", "csubj:pass"}
                )
            else:
                feats["head_has_overt_nsubj"] = False
                feats["head_has_expl"] = False
                feats["head_has_csubj"] = False

            # --- Reachable subject: walk up dep chain ---
            # For aux/cop the subject is on the head; for conj it's on
            # the first conjunct; for xcomp it's inherited.  Walk up to
            # 3 hops looking for any node that carries a subject dep.
            feats["has_reachable_subject"] = self._has_reachable_subject(tok)

            rows.append(
                VerbFeatureRow(
                    token_idx=tok.i,
                    char_offset=tok.idx,
                    verb_text=tok.text,
                    verb_lemma=lemma,
                    person=person,
                    number=number,
                    morph_label_suffix=morph_suffix,
                    features=feats,
                )
            )

        return rows

    def extract_doc(
        self,
        doc: spacy.tokens.Doc,
        genre: str = "unknown",
    ) -> List[VerbFeatureRow]:
        """Extract feature rows for all finite verbs in a Doc."""
        rows: List[VerbFeatureRow] = []
        for sent in doc.sents:
            rows.extend(self.extract_sentence(sent, genre=genre))
        return rows

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _index_by_token_idx(
        records: List[Dict[str, Any]],
    ) -> Dict[int, Dict[str, Any]]:
        """Index list of annotation dicts by 'token_idx' key."""
        return {r["token_idx"]: r for r in records if "token_idx" in r}

    @staticmethod
    def _index_by_verb_idx(
        records: List[Dict[str, Any]],
    ) -> Dict[int, Dict[str, Any]]:
        """Index list of annotation dicts by 'verb_idx' key."""
        return {r["verb_idx"]: r for r in records if "verb_idx" in r}

    @staticmethod
    def _verb_is_head_of_wh(
        verb_tok: spacy.tokens.Token,
        wh_annot: Dict[str, Any],
        sent: spacy.tokens.Span,
    ) -> bool:
        """Check if verb is the head of the wh-word."""
        wh_idx = wh_annot.get("wh_idx")
        if wh_idx is None:
            return False
        abs_idx = wh_idx + sent.start
        if abs_idx >= len(verb_tok.doc):
            return False
        wh_tok = verb_tok.doc[abs_idx]
        return wh_tok.head == verb_tok

    @staticmethod
    def _get_wh_extraction_type(
        verb_tok: spacy.tokens.Token,
        wh_extractions: List[Dict[str, Any]],
        sent: spacy.tokens.Span,
    ) -> str:
        """Get extraction type for wh-words headed by this verb."""
        for wh in wh_extractions:
            wh_idx = wh.get("wh_idx")
            if wh_idx is None:
                continue
            abs_idx = wh_idx + sent.start
            if abs_idx >= len(verb_tok.doc):
                continue
            wh_tok = verb_tok.doc[abs_idx]
            if wh_tok.head == verb_tok:
                return wh.get("extraction_type", "none")
        return "none"

    @staticmethod
    def _negation_heads_verb(
        neg_annot: Dict[str, Any],
        verb_tok: spacy.tokens.Token,
        sent: spacy.tokens.Span,
    ) -> bool:
        """Check if negation token's head is this verb."""
        neg_idx = neg_annot.get("token_idx")
        if neg_idx is None:
            return False
        abs_idx = neg_idx + sent.start
        if abs_idx >= len(verb_tok.doc):
            return False
        neg_tok = verb_tok.doc[abs_idx]
        return neg_tok.head == verb_tok

    _SUBJECT_DEPS = frozenset({"nsubj", "nsubj:pass", "csubj", "csubj:pass", "expl"})
    _CHAIN_DEPS = frozenset({"aux", "aux:pass", "cop", "conj", "xcomp"})

    # POS tags for impersonal predicate heads (è + ADJ/NOUN)
    _IMPERSONAL_HEAD_POS = frozenset({"ADJ", "NOUN"})

    @classmethod
    def _is_copular_impersonal(
        cls, tok: spacy.tokens.Token, person: str | None,
    ) -> bool:
        """Detect copular impersonal constructions.

        True for 3rd-person copula verbs whose head is an adjective or noun
        with no overt subject, e.g. "è importante che...", "è un peccato".
        Also catches aux verbs in compound impersonal copulas
        ("è stato importante").
        """
        if person != "3":
            return False

        # Direct cop: "è" (cop) → "importante" (ADJ, ROOT)
        if tok.dep_ == "cop":
            head = tok.head
            if head.pos_ in cls._IMPERSONAL_HEAD_POS:
                head_child_deps = {c.dep_ for c in head.children}
                if not (head_child_deps & {"nsubj", "nsubj:pass"}):
                    return True

        # Aux of a cop: "è" (aux) → "stato" (cop) → "importante" (ADJ)
        if tok.dep_ in ("aux", "aux:pass") and tok.head.dep_ == "cop":
            predicate = tok.head.head
            if predicate.pos_ in cls._IMPERSONAL_HEAD_POS:
                pred_child_deps = {c.dep_ for c in predicate.children}
                if not (pred_child_deps & {"nsubj", "nsubj:pass"}):
                    return True

        return False

    @classmethod
    def _has_reachable_subject(cls, tok: spacy.tokens.Token) -> bool:
        """Walk up the dependency chain looking for any subject.

        Checks the verb itself, then follows aux/cop/conj/xcomp links
        up to 3 hops, checking each node's children for a subject dep.
        """
        current = tok
        visited = {tok.i}
        for _ in range(4):  # current node + up to 3 hops
            child_deps = {c.dep_ for c in current.children}
            if child_deps & cls._SUBJECT_DEPS:
                return True
            # Only follow the chain for specific dep relations
            if current.dep_ in cls._CHAIN_DEPS and current.head.i not in visited:
                visited.add(current.head.i)
                current = current.head
            else:
                break
        return False

    @staticmethod
    def _distance_to_root(tok: spacy.tokens.Token) -> int:
        """Count hops from token to ROOT in dependency tree."""
        dist = 0
        current = tok
        visited = set()
        while current.dep_ != "ROOT" and current.i not in visited:
            visited.add(current.i)
            current = current.head
            dist += 1
        return dist


# Feature names for consistent ordering
FEATURE_NAMES = [
    "verb_tense",
    "verb_mood",
    "verb_aspect",
    "verb_person",
    "verb_number",
    "verb_clause_position",
    "clause_type",
    "dep_label",
    "clause_subject_status",
    "clause_is_finite",
    "transitivity",
    "object_status",
    "iobject_status",
    "argstruct_subject_status",
    "has_clitic_on_verb",
    "has_reflexive_clitic",
    "n_clitics",
    "is_weather_verb",
    "is_impersonal_verb",
    "is_necessity_verb",
    "is_copular_impersonal",
    "n_clauses",
    "max_embedding_depth",
    "has_coordination",
    "is_fragment",
    "is_imperative",
    "is_in_wh_clause",
    "wh_extraction_type",
    "is_in_relcl",
    "rel_type",
    "rel_pronoun_is_subject",
    "is_bridge_complement",
    "has_complementizer",
    "has_negation",
    "negation_position",
    "prev_root_subject_person",
    "topic_matches_verb_morph",
    "verb_position_normalized",
    "is_sentence_initial_verb",
    "distance_to_root",
    "is_essere",
    "is_avere",
    "is_modal",
    "has_overt_nsubj",
    "has_expl",
    "has_csubj",
    "is_xcomp",
    "is_aux_or_cop",
    "head_has_overt_nsubj",
    "head_has_expl",
    "head_has_csubj",
    "has_reachable_subject",
]
