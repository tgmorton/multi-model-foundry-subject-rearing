#!/usr/bin/env python3
"""Post-process R1 pronoun recovery annotations.

Fixes:
1. CONJ labels → resolve to person/number via spaCy verb morphology
2. Invalid labels (PRO.2sg:lei → PRO.3sg:lei, ella → lei)
3. Flags text corruption (annotated text differs from original)
4. Cross-checks ALL marker labels against spaCy verb morphology

Usage:
    python scripts/postprocess_r1_annotations.py \
        --input data/pronoun_recovery/annotations/it/r1_checkpoint.jsonl \
        --output data/pronoun_recovery/annotations/it/r1_postprocessed.jsonl
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import spacy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Valid label → lexical form mappings for Italian
PERSON_NUMBER_MAP = {
    ("1", "Sing"): ("PRO.1sg", "io"),
    ("2", "Sing"): ("PRO.2sg", "tu"),
    ("3", "Sing"): ("PRO.3sg", None),  # lui/lei/esso/essa resolved separately
    ("1", "Plur"): ("PRO.1pl", "noi"),
    ("2", "Plur"): ("PRO.2pl", "voi"),
    ("3", "Plur"): ("PRO.3pl", "loro"),
}

# For 3sg, keep original lexical form if valid, otherwise default to lui/lei
VALID_3SG_FORMS = {"lui", "lei", "esso", "essa"}

MARKER_PATTERN = re.compile(r"\[PRO\.[\w.]+(?::[\w]+)?\]|\[CONJ\]")

# Pattern to find each marker and capture the next few words after it
MARKER_CONTEXT_PATTERN = re.compile(
    r"(\[(?:PRO\.[\w.]+(?::[\w]+)?|CONJ)\])\s*((?:\S+\s*){1,6})"
)

# Italian clitics/particles that can appear between subject pronoun and verb
ITALIAN_CLITICS = {
    "mi", "ti", "ci", "vi", "si", "lo", "la", "li", "le", "gli", "ne",
    "me", "te", "ce", "ve", "se", "l", "non", "ne", "glielo", "gliela",
    "glieli", "gliele", "gliene", "anche", "già", "gia", "più", "piu",
    "poi", "ancora", "sempre", "quindi", "però", "pero", "tuttavia",
}


def detect_text_corruption(original_text: str, annotated_text: str) -> bool:
    """Check if R1 modified the original text beyond inserting markers."""
    stripped = MARKER_PATTERN.sub("", annotated_text)
    orig_norm = " ".join(original_text.split())
    stripped_norm = " ".join(stripped.split())
    return orig_norm != stripped_norm


def find_verb_after_marker(nlp, annotated_text: str, marker_match) -> "spacy.tokens.Token | None":
    """Find the finite verb in the words immediately following a marker tag.

    Works directly on the annotated text — no position alignment needed.
    Extracts the words after the marker, parses them with spaCy, and returns
    the first finite verb token.
    """
    # Get the text captured after the marker (up to 6 words)
    after_text = marker_match.group(2).strip()
    if not after_text:
        return None

    # Strip any nested markers that might appear in the captured text
    after_clean = MARKER_PATTERN.sub("", after_text).strip()
    if not after_clean:
        return None

    doc = nlp(after_clean)
    for tok in doc:
        morph = tok.morph
        verb_form = morph.get("VerbForm")
        if verb_form and "Fin" in str(verb_form):
            return tok
    return None


def resolve_conj_label(tok: "spacy.tokens.Token") -> "tuple[str, str] | None":
    """Resolve a CONJ marker to a proper person/number label using verb morphology."""
    morph = tok.morph
    person = morph.get("Person")
    number = morph.get("Number")

    if not person or not number:
        return None

    p = person[0]
    n = number[0]
    key = (p, n)

    if key not in PERSON_NUMBER_MAP:
        return None

    label, lex = PERSON_NUMBER_MAP[key]
    if label == "PRO.3sg":
        lex = "lui"  # default for CONJ resolution; context-dependent
    return label, lex


def check_marker_morphology(tok: "spacy.tokens.Token", marker_label: str) -> "tuple[str, str] | None":
    """Cross-check a marker's label against the verb's morphology.

    Returns (correct_label, correct_lex) if mismatch detected, None if OK.
    """
    morph = tok.morph
    person = morph.get("Person")
    number = morph.get("Number")

    if not person or not number:
        return None

    p = person[0]
    n = number[0]
    key = (p, n)

    if key not in PERSON_NUMBER_MAP:
        return None

    expected_label, expected_lex = PERSON_NUMBER_MAP[key]

    # Check if current label matches expected
    if marker_label == expected_label:
        return None  # OK

    # Mismatch detected
    if expected_label == "PRO.3sg":
        expected_lex = "lui"
    return expected_label, expected_lex


def fix_invalid_labels(markers: list) -> tuple[list, int]:
    """Fix known invalid label combinations."""
    fixed = 0
    for m in markers:
        label = m.get("label", "")
        lex = m.get("lexical_form", "")

        # PRO.2sg:lei → PRO.3sg:lei (formal address is grammatically 3sg)
        if label == "PRO.2sg" and lex == "lei":
            m["label"] = "PRO.3sg"
            fixed += 1

        # ella → lei
        if lex == "ella":
            m["lexical_form"] = "lei"
            fixed += 1

        # egli → lui
        if lex == "egli":
            m["lexical_form"] = "lui"
            fixed += 1

    return markers, fixed


def rebuild_annotated_text(original_text: str, markers: list) -> str:
    """Rebuild annotated_text by inserting markers into the original text."""
    # Sort markers by position (descending) so insertions don't shift positions
    sorted_markers = sorted(markers, key=lambda m: m["position"], reverse=True)
    result = original_text
    for m in sorted_markers:
        pos = m["position"]
        label = m["label"]
        lex = m.get("lexical_form")
        if lex:
            tag = f"[{label}:{lex}]"
        else:
            tag = f"[{label}]"
        result = result[:pos] + tag + " " + result[pos:]
    return result


def main():
    parser = argparse.ArgumentParser(description="Post-process R1 annotations")
    parser.add_argument("--input", required=True, help="Input R1 checkpoint JSONL")
    parser.add_argument("--output", required=True, help="Output post-processed JSONL")
    parser.add_argument("--corrupted-ids", default=None,
                        help="Output file for corrupted example IDs (for re-annotation)")
    parser.add_argument("--spacy-model", default="it_core_news_lg",
                        help="spaCy model for morphological analysis")
    parser.add_argument("--fix-morphology", action="store_true",
                        help="Also cross-check and fix non-CONJ marker labels against verb morphology")
    args = parser.parse_args()

    log.info("Loading spaCy model %s...", args.spacy_model)
    nlp = spacy.load(args.spacy_model)

    # Stats
    total = 0
    corrupted = 0
    conj_resolved = 0
    conj_unresolved = 0
    invalid_fixed = 0
    morph_mismatches = 0
    morph_fixed = 0
    clean_count = 0
    corrupted_ids = []

    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            obj = json.loads(line)
            total += 1

            original_text = obj.get("original_text", "")
            annotated_text = obj.get("annotated_text", "")
            markers = obj.get("markers", [])

            # 1. Check text corruption
            is_corrupted = detect_text_corruption(original_text, annotated_text)
            if is_corrupted:
                corrupted += 1
                corrupted_ids.append(obj["id"])
                obj["postprocess_status"] = "corrupted"
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            # 2. Fix invalid labels
            markers, n_fixed = fix_invalid_labels(markers)
            invalid_fixed += n_fixed

            # 3. Find all markers in annotated text and their following words
            #    Build a map: marker_tag_string → verb token from spaCy
            marker_verbs = {}
            for mm in MARKER_CONTEXT_PATTERN.finditer(annotated_text):
                tag = mm.group(1)       # e.g. "[PRO.CONJ]" or "[PRO.1sg:io]"
                tok = find_verb_after_marker(nlp, annotated_text, mm)
                # Use the match start position as key (unique per marker occurrence)
                marker_verbs[mm.start()] = tok

            # 4. Match markers list entries to their annotated-text positions
            #    Walk through annotated_text finding each marker tag in order
            marker_positions_in_annotated = []
            search_start = 0
            for m in markers:
                label = m.get("label", "")
                lex = m.get("lexical_form")
                # Build possible tag variants (annotated text uses PRO. prefix)
                candidates = []
                if lex:
                    candidates.append(f"[{label}:{lex}]")
                    candidates.append(f"[PRO.{label}:{lex}]")
                else:
                    candidates.append(f"[{label}]")
                    candidates.append(f"[PRO.{label}]")
                idx = -1
                matched_tag = ""
                for tag in candidates:
                    idx = annotated_text.find(tag, search_start)
                    if idx >= 0:
                        matched_tag = tag
                        break
                marker_positions_in_annotated.append(idx)
                if idx >= 0:
                    search_start = idx + len(matched_tag)

            # 5. Process each marker
            new_markers = []
            for i, m in enumerate(markers):
                label = m.get("label", "")
                lex = m.get("lexical_form")
                ann_pos = marker_positions_in_annotated[i]

                # Find the closest MARKER_CONTEXT_PATTERN match position
                tok = None
                if ann_pos >= 0:
                    # Find the marker_verbs entry whose start is closest to ann_pos
                    best_key = None
                    best_dist = float("inf")
                    for key in marker_verbs:
                        dist = abs(key - ann_pos)
                        if dist < best_dist:
                            best_dist = dist
                            best_key = key
                    if best_key is not None and best_dist <= 5:
                        tok = marker_verbs[best_key]

                if label == "CONJ":
                    if tok:
                        resolved = resolve_conj_label(tok)
                        if resolved:
                            new_label, new_lex = resolved
                            m["label"] = new_label
                            m["lexical_form"] = new_lex
                            m["conj_resolved_from"] = tok.text
                            conj_resolved += 1
                        else:
                            conj_unresolved += 1
                            m["conj_resolved_from"] = f"FAILED:{tok.text}"
                    else:
                        conj_unresolved += 1
                        m["conj_resolved_from"] = "NO_VERB_FOUND"

                elif args.fix_morphology and label.startswith("PRO."):
                    if tok:
                        fix = check_marker_morphology(tok, label)
                        if fix:
                            morph_mismatches += 1
                            new_label, new_lex = fix
                            m["morph_suggestion"] = {
                                "verb": tok.text,
                                "suggested_label": new_label,
                                "suggested_lex": new_lex,
                                "original_label": label,
                            }

                new_markers.append(m)

            obj["markers"] = new_markers

            # 6. Rebuild annotated_text and clean_text from original + fixed markers
            obj["annotated_text"] = rebuild_annotated_text(original_text, new_markers)
            obj["clean_text"] = original_text  # clean = original for non-corrupted
            obj["is_valid"] = True
            obj["postprocess_status"] = "ok"
            clean_count += 1

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

            if total % 500 == 0:
                log.info(
                    "[%d] corrupted=%d conj_resolved=%d conj_unresolved=%d invalid_fixed=%d morph_mismatches=%d",
                    total, corrupted, conj_resolved, conj_unresolved, invalid_fixed, morph_mismatches,
                )

    # Write corrupted IDs
    if args.corrupted_ids and corrupted_ids:
        with open(args.corrupted_ids, "w") as f:
            for cid in corrupted_ids:
                f.write(cid + "\n")
        log.info("Wrote %d corrupted IDs to %s", len(corrupted_ids), args.corrupted_ids)

    log.info("=" * 60)
    log.info("Post-processing complete!")
    log.info("  Total:            %d", total)
    log.info("  Clean:            %d (%.1f%%)", clean_count, 100 * clean_count / total)
    log.info("  Corrupted:        %d (%.1f%%)", corrupted, 100 * corrupted / total)
    log.info("  CONJ resolved:    %d", conj_resolved)
    log.info("  CONJ unresolved:  %d", conj_unresolved)
    log.info("  Invalid fixed:    %d", invalid_fixed)
    log.info("  Morph mismatches: %d (logged, not auto-fixed)", morph_mismatches)


if __name__ == "__main__":
    main()
