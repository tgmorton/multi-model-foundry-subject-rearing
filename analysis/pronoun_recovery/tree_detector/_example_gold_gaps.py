"""Pull example sentences for each gold data gap category.

Processes aligned_checkpoint.jsonl with spaCy, identifies verbs matching
each gold gap pattern, and prints glossed examples.
"""

import json
from pathlib import Path
from collections import defaultdict

import spacy

from analysis.corpus_descriptives.constants import IMPERSONAL_VERBS_IT, WEATHER_VERBS_IT, NECESSITY_VERBS_IT
from analysis.pronoun_recovery.constants import MORPH_TO_LABEL_SUFFIX

DATA_PATH = Path("data/pronoun_recovery/europarl_aligned/it/aligned_checkpoint.jsonl")

# How many examples per category
N_EXAMPLES = 3

# Modal lemmas
_MODAL_LEMMAS = frozenset({"potere", "dovere", "volere", "sapere"})


def _person_number(tok):
    p = tok.morph.get("Person")
    n = tok.morph.get("Number")
    return (p[0] if p else None, n[0] if n else None)


def _is_finite(tok):
    vf = tok.morph.get("VerbForm")
    return tok.pos_ in ("VERB", "AUX") and vf and vf[0] == "Fin"


def _has_marker(tok, markers):
    """Check if this token has a gold marker within 3 chars."""
    for m in markers:
        if abs(tok.idx - m["position"]) <= 3:
            return True
    return False


def _subject_deps():
    return {"nsubj", "nsubj:pass", "csubj", "csubj:pass", "expl"}


def categorize_verb(tok, markers, person, number):
    """Categorize an unlabeled finite verb into a gold gap type (or None)."""
    dep = tok.dep_
    lemma = tok.lemma_.lower()
    p = person

    if _has_marker(tok, markers):
        return None  # has a marker, not a gap

    if not _is_finite(tok):
        return None

    # Only 1st/2nd person for gold gaps
    if p not in ("1", "2"):
        return None

    # Must not have an overt subject
    child_deps = {c.dep_ for c in tok.children}
    if child_deps & _subject_deps():
        return None

    # Categorize
    if dep in ("aux", "aux:pass"):
        if lemma in _MODAL_LEMMAS:
            return "gold_gap_1p2p_modal_aux"
        elif lemma == "avere":
            return "gold_gap_1p2p_avere_aux"
        elif lemma == "essere":
            return "gold_gap_1p2p_essere_aux"
        else:
            return "gold_gap_1p2p_other_aux"
    elif dep == "cop":
        return "gold_gap_1p2p_cop"
    elif dep == "ROOT":
        return "gold_gap_1p2p_root"
    elif dep == "conj":
        return "gold_gap_1p2p_conj"
    elif dep in ("advcl", "ccomp"):
        return "gold_gap_1p2p_subordinate"
    elif dep in ("acl:relcl", "acl"):
        return "gold_gap_1p2p_relcl"
    else:
        return "gold_gap_1p2p_other_dep"


def _morph_label(person, number):
    suffix = MORPH_TO_LABEL_SUFFIX.get((person, number))
    return f"PRO.{suffix}" if suffix else "PRO.?"


def _gloss_person(person, number):
    """Simple English pronoun gloss."""
    gloss = {
        ("1", "Sing"): "I",
        ("1", "Plur"): "we",
        ("2", "Sing"): "you(sg)",
        ("2", "Plur"): "you(pl)",
        ("3", "Sing"): "he/she/it",
        ("3", "Plur"): "they",
    }
    return gloss.get((person, number), "?")


def main():
    print("Loading spaCy model...")
    nlp = spacy.load("it_core_news_sm")

    print(f"Loading records from {DATA_PATH}...")
    records = []
    with open(DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Processing {len(records)} records...\n")

    examples = defaultdict(list)
    texts = [r["clean_text"] for r in records]

    for record, doc in zip(records, nlp.pipe(texts, batch_size=50)):
        markers = record.get("markers", [])

        for tok in doc:
            person, number = _person_number(tok)
            if person is None:
                continue

            cat = categorize_verb(tok, markers, person, number)
            if cat is None:
                continue

            if len(examples[cat]) >= N_EXAMPLES:
                continue

            # Get surrounding context
            sent = tok.sent
            sent_text = sent.text.strip()

            # Mark the verb in the sentence
            start = tok.idx - sent.start_char
            end = start + len(tok.text)
            marked = sent_text[:start] + f"[{tok.text}]" + sent_text[end:]

            # Get the marker labels for this sentence
            sent_markers = [
                m for m in markers
                if sent.start_char <= m["position"] < sent.end_char
            ]
            marker_labels = [m["label"] for m in sent_markers]

            pronoun = _gloss_person(person, number)

            examples[cat].append({
                "italian": marked,
                "verb": tok.text,
                "lemma": tok.lemma_,
                "dep": tok.dep_,
                "person_number": f"{person}{number[0] if number else '?'}",
                "dropped_pronoun": pronoun,
                "existing_markers": marker_labels or ["none"],
                "head": f"{tok.head.text} ({tok.head.dep_})",
            })

        # Stop early if we have enough examples for all categories
        all_cats = [
            "gold_gap_1p2p_modal_aux", "gold_gap_1p2p_avere_aux",
            "gold_gap_1p2p_essere_aux", "gold_gap_1p2p_other_aux",
            "gold_gap_1p2p_cop", "gold_gap_1p2p_root",
            "gold_gap_1p2p_conj", "gold_gap_1p2p_subordinate",
            "gold_gap_1p2p_relcl", "gold_gap_1p2p_other_dep",
        ]
        if all(len(examples[c]) >= N_EXAMPLES for c in all_cats):
            break

    # Print examples
    for cat in sorted(examples.keys()):
        print(f"\n{'='*70}")
        print(f"  {cat} ({len(examples[cat])} examples)")
        print(f"{'='*70}")
        for i, ex in enumerate(examples[cat], 1):
            print(f"\n  Example {i}:")
            print(f"    IT: {ex['italian']}")
            print(f"    Verb: {ex['verb']} ({ex['lemma']}, {ex['dep']}, {ex['person_number']})")
            print(f"    Dropped: [{ex['dropped_pronoun']}] {ex['verb']}")
            print(f"    Head: {ex['head']}")
            print(f"    Existing markers in sentence: {ex['existing_markers']}")


if __name__ == "__main__":
    main()
