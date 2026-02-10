#!/usr/bin/env python3
"""Merge re-annotated corrupted examples into the postprocessed file.

1. Reads the postprocessed file (has corrupted examples flagged)
2. Reads the re-annotated file (R1 re-run on corrupted examples)
3. Post-processes the re-annotated examples (CONJ resolution, invalid labels)
4. Replaces corrupted entries with the re-annotated versions
5. Writes the final merged output

Usage:
    python scripts/merge_postprocessed.py \
        --postprocessed data/pronoun_recovery/annotations/it/r1_postprocessed.jsonl \
        --reannotated /tmp/r1_corrupted_redo.jsonl \
        --output data/pronoun_recovery/annotations/it/r1_final.jsonl
"""

import argparse
import json
import logging
import re
import sys

import spacy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MARKER_PATTERN = re.compile(r"\[PRO\.[\w.]+(?::[\w]+)?\]|\[CONJ\]")
MARKER_CONTEXT_PATTERN = re.compile(
    r"(\[(?:PRO\.[\w.]+(?::[\w]+)?|CONJ)\])\s*((?:\S+\s*){1,6})"
)

PERSON_NUMBER_MAP = {
    ("1", "Sing"): ("PRO.1sg", "io"),
    ("2", "Sing"): ("PRO.2sg", "tu"),
    ("3", "Sing"): ("PRO.3sg", None),
    ("1", "Plur"): ("PRO.1pl", "noi"),
    ("2", "Plur"): ("PRO.2pl", "voi"),
    ("3", "Plur"): ("PRO.3pl", "loro"),
}


def detect_text_corruption(original_text, annotated_text):
    stripped = MARKER_PATTERN.sub("", annotated_text)
    return " ".join(original_text.split()) != " ".join(stripped.split())


def find_verb_after_marker(nlp, annotated_text, mm):
    after_text = mm.group(2).strip()
    if not after_text:
        return None
    after_clean = MARKER_PATTERN.sub("", after_text).strip()
    if not after_clean:
        return None
    doc = nlp(after_clean)
    for tok in doc:
        vf = tok.morph.get("VerbForm")
        if vf and "Fin" in str(vf):
            return tok
    return None


def resolve_conj_label(tok):
    person = tok.morph.get("Person")
    number = tok.morph.get("Number")
    if not person or not number:
        return None
    key = (person[0], number[0])
    if key not in PERSON_NUMBER_MAP:
        return None
    label, lex = PERSON_NUMBER_MAP[key]
    if label == "PRO.3sg":
        lex = "lui"
    return label, lex


def fix_invalid_labels(markers):
    fixed = 0
    for m in markers:
        label = m.get("label", "")
        lex = m.get("lexical_form", "")
        if label == "PRO.2sg" and lex == "lei":
            m["label"] = "PRO.3sg"
            fixed += 1
        if lex == "ella":
            m["lexical_form"] = "lei"
            fixed += 1
        if lex == "egli":
            m["lexical_form"] = "lui"
            fixed += 1
    return markers, fixed


def rebuild_annotated_text(original_text, markers):
    sorted_markers = sorted(markers, key=lambda m: m["position"], reverse=True)
    result = original_text
    for m in sorted_markers:
        pos = m["position"]
        label = m["label"]
        lex = m.get("lexical_form")
        tag = f"[{label}:{lex}]" if lex else f"[{label}]"
        result = result[:pos] + tag + " " + result[pos:]
    return result


def postprocess_one(obj, nlp):
    """Post-process a single re-annotated example. Returns (obj, still_corrupted)."""
    original_text = obj.get("original_text", "")
    annotated_text = obj.get("annotated_text", "")
    markers = obj.get("markers", [])

    if detect_text_corruption(original_text, annotated_text):
        obj["postprocess_status"] = "corrupted"
        return obj, True

    markers, _ = fix_invalid_labels(markers)

    # Resolve CONJ markers
    marker_verbs = {}
    for mm in MARKER_CONTEXT_PATTERN.finditer(annotated_text):
        tok = find_verb_after_marker(nlp, annotated_text, mm)
        marker_verbs[mm.start()] = tok

    # Match markers to annotated text positions
    search_start = 0
    marker_ann_positions = []
    for m in markers:
        label = m.get("label", "")
        lex = m.get("lexical_form")
        candidates = []
        if lex:
            candidates.extend([f"[{label}:{lex}]", f"[PRO.{label}:{lex}]"])
        else:
            candidates.extend([f"[{label}]", f"[PRO.{label}]"])
        idx = -1
        matched_tag = ""
        for tag in candidates:
            idx = annotated_text.find(tag, search_start)
            if idx >= 0:
                matched_tag = tag
                break
        marker_ann_positions.append(idx)
        if idx >= 0:
            search_start = idx + len(matched_tag)

    for i, m in enumerate(markers):
        label = m.get("label", "")
        ann_pos = marker_ann_positions[i]

        if label == "CONJ" and ann_pos >= 0:
            best_key = min(marker_verbs, key=lambda k: abs(k - ann_pos), default=None)
            tok = marker_verbs.get(best_key) if best_key is not None and abs(best_key - ann_pos) <= 5 else None
            if tok:
                resolved = resolve_conj_label(tok)
                if resolved:
                    m["label"], m["lexical_form"] = resolved
                    m["conj_resolved_from"] = tok.text

    obj["markers"] = markers
    obj["annotated_text"] = rebuild_annotated_text(original_text, markers)
    obj["clean_text"] = original_text
    obj["is_valid"] = True
    obj["postprocess_status"] = "ok"
    return obj, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--postprocessed", required=True)
    parser.add_argument("--reannotated", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--spacy-model", default="it_core_news_lg")
    args = parser.parse_args()

    log.info("Loading spaCy model %s...", args.spacy_model)
    nlp = spacy.load(args.spacy_model)

    # Load re-annotated examples
    reannotated = {}
    with open(args.reannotated) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("status") == "success":
                reannotated[obj["id"]] = obj
    log.info("Loaded %d re-annotated examples", len(reannotated))

    total = 0
    replaced = 0
    still_corrupted = 0
    clean = 0

    with open(args.postprocessed) as fin, open(args.output, "w") as fout:
        for line in fin:
            obj = json.loads(line)
            total += 1
            eid = obj["id"]

            if obj.get("postprocess_status") == "corrupted" and eid in reannotated:
                # Replace with re-annotated version
                new_obj = reannotated[eid]
                new_obj, is_corrupt = postprocess_one(new_obj, nlp)
                if is_corrupt:
                    still_corrupted += 1
                else:
                    replaced += 1
                    clean += 1
                fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
            else:
                if obj.get("postprocess_status") != "corrupted":
                    clean += 1
                fout.write(line)

            if total % 2000 == 0:
                log.info("[%d] replaced=%d still_corrupted=%d", total, replaced, still_corrupted)

    log.info("=" * 60)
    log.info("Merge complete!")
    log.info("  Total:           %d", total)
    log.info("  Clean:           %d (%.1f%%)", clean, 100 * clean / total)
    log.info("  Replaced:        %d", replaced)
    log.info("  Still corrupted: %d", still_corrupted)


if __name__ == "__main__":
    main()
