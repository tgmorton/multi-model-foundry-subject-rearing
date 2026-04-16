#!/usr/bin/env python3
"""Build a Spanish null-subject gold annotation sample from train_90M.

Produces a single JSONL file at ``data/spanish/gold/stimuli.jsonl`` ready
for upload to the annotation web app via ``POST /api/stimuli/load``.

Sampling design
---------------
- **IID** (500): 50 sentences per source, uniform random. Reflects
  natural distribution within each register.
- **Stratified** (500): 25 ``predicted_null`` + 25 ``predicted_overt`` per
  source. Over-represents positives for per-category F1 diagnostics.
- **IAA** (100): random 100 from the IID ∪ stratified union, flagged
  via ``metadata.sample_types`` for double annotation.

A sentence that lands in both IID and stratified is emitted once with
``sample_types: ["iid", "stratified"]``.

Pre-labelling uses spaCy ``es_core_news_lg`` with the five-way
subject-status classifier from
``analysis/pronoun_recovery/parallel_data/it_null_subject_detector.py``
(UD deprels are language-agnostic, so the Italian classifier works for
Spanish at ~80% accuracy — more than enough for stratification).

Usage
-----
    python scripts/build_spanish_gold_sample.py \
        --source_dir data/spanish/train_90M \
        --output_dir data/spanish/gold

    # Dry run on a single source:
    python scripts/build_spanish_gold_sample.py \
        --source_dir data/spanish/train_90M \
        --output_dir data/spanish/gold \
        --only_source qed
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

RANDOM_SEED = 42
POOL_SIZE_PER_SOURCE = 3000
MIN_TOKENS = 5
MAX_TOKENS = 40
CONTEXT_BEFORE_N = 2
CONTEXT_AFTER_N = 1

IID_PER_SOURCE = 50
STRATIFIED_PER_CLASS_PER_SOURCE = 25
IAA_TOTAL = 100


@dataclass
class Candidate:
    """A parsed candidate sentence with its predicted null-subject flag."""

    source: str            # e.g. "qed"
    line_idx: int          # 0-based index into source_lines[source]
    text: str              # cleaned, length-filtered sentence
    predicted_null: bool
    predicted_verbs: List[Dict] = field(default_factory=list)


# ── Line cleaning ────────────────────────────────────────────────────

# Speaker tag patterns reused from
# analysis/pronoun_recovery/annotation/sampler.py:24-42
import re  # noqa: E402

_CHILDES_PATTERN = re.compile(r"^\*[A-Z]{2,4}:\t(.*)$")
_SWITCHBOARD_PATTERN = re.compile(r"^[AB]:\t(.*)$")


def _clean_line(text: str, source: str) -> str:
    """Strip CHILDES/Switchboard speaker tags; pass through otherwise."""
    src = source.lower()
    if src == "childes":
        m = _CHILDES_PATTERN.match(text)
        if m:
            return m.group(1).strip()
    elif src == "switchboard":
        m = _SWITCHBOARD_PATTERN.match(text)
        if m:
            return m.group(1).strip()
    return text.strip()


def _length_ok(text: str) -> bool:
    n = len(text.split())
    return MIN_TOKENS <= n <= MAX_TOKENS


# ── Finite-verb classification ───────────────────────────────────────

_OVERT_SUBJECT_DEPS = {"nsubj", "nsubj:pass", "nsubjpass"}
_CLAUSAL_SUBJECT_DEPS = {"csubj", "csubj:pass", "csubjpass"}
_ANY_SUBJECT_DEPS = _OVERT_SUBJECT_DEPS | _CLAUSAL_SUBJECT_DEPS | {"expl"}
_AUX_COP_DEPS = {"aux", "aux:pass", "auxpass", "cop"}


def _effective_subject_deps(tok) -> set:
    """Return the set of dep labels attached to this verb's *effective*
    subject slot, walking through aux/cop chains.

    In Spanish UD:
      - Passives: "fueron diseñadas" → `fueron` is AUX (dep=aux) of
        `diseñadas` (ROOT VERB). `nsubj` attaches to `diseñadas`, not
        `fueron`. But `fueron` is the finite verb.
      - Copulas: "Juan es feliz" → `es` is AUX (dep=cop) of `feliz`
        (ROOT ADJ). `nsubj` attaches to `feliz`.

    So when classifying a finite AUX, look at the HEAD's children too.
    Also handle the reverse case: a finite verb whose aux/cop child
    "steals" no subjects — we include the verb's own children plus any
    via aux/cop head-chain up to 2 hops.
    """
    # Direct children of this token.
    deps = {c.dep_ for c in tok.children}

    # If THIS token is itself an aux/cop, also check its head (the
    # content word) — that's where nsubj usually attaches.
    if tok.dep_ in _AUX_COP_DEPS:
        deps.update(c.dep_ for c in tok.head.children)

    return deps


def classify_finite_verbs(doc) -> List[Dict]:
    """Classify each finite verb's subject status.

    Logic mirrors
    ``analysis/pronoun_recovery/parallel_data/it_null_subject_detector.py``
    (UD deprels are language-agnostic) with a Spanish-specific extension:
    aux/cop finite verbs check their HEAD's children for the subject,
    since Spanish UD attaches nsubj to the content head (participle in
    passives, adjective in copulas).

    Returns a list of dicts, one per finite verb, with ``text``,
    ``lemma``, ``subject_status``, ``person``, ``number``.
    """
    verbs: List[Dict] = []
    for tok in doc:
        if tok.pos_ not in ("VERB", "AUX"):
            continue
        verb_forms = tok.morph.get("VerbForm")
        if not verb_forms or verb_forms[0] != "Fin":
            continue

        dep_set = _effective_subject_deps(tok)

        if "expl" in dep_set:
            status = "expletive"
        elif dep_set & _OVERT_SUBJECT_DEPS:
            status = "overt"
        elif dep_set & _CLAUSAL_SUBJECT_DEPS:
            status = "clausal"
        elif tok.dep_ == "xcomp":
            head_deps = {c.dep_ for c in tok.head.children}
            status = "inherited" if head_deps & _ANY_SUBJECT_DEPS else "null"
        else:
            status = "null"

        person = tok.morph.get("Person")
        number = tok.morph.get("Number")
        verbs.append({
            "text": tok.text,
            "lemma": tok.lemma_,
            "subject_status": status,
            "person": person[0] if person else None,
            "number": number[0] if number else None,
        })
    return verbs


# ── Pool construction ────────────────────────────────────────────────

def load_source_lines(train_path: Path) -> List[str]:
    """Read a train file, return ALL cleaned lines (not length-filtered).

    Preserves 1-to-1 indexing with the file so context lookups work.
    Blank or placeholder lines become empty strings at the same index.
    """
    source = train_path.stem
    out: List[str] = []
    with open(train_path, "r", encoding="utf-8") as f:
        for raw in f:
            if not raw.strip() or raw.startswith("#"):
                out.append("")
                continue
            out.append(_clean_line(raw, source))
    return out


def build_candidate_pool(
    source: str,
    lines: List[str],
    nlp,
    rng: random.Random,
) -> List[Candidate]:
    """Parse up to POOL_SIZE_PER_SOURCE length-filtered lines from this source.

    Random-shuffles eligible line indices, takes the first N, parses them
    in a batch, classifies finite verbs, returns candidates.
    """
    eligible_idxs = [i for i, l in enumerate(lines) if _length_ok(l)]
    if not eligible_idxs:
        log.warning("  %s: no length-eligible lines", source)
        return []

    rng.shuffle(eligible_idxs)
    chosen = eligible_idxs[:POOL_SIZE_PER_SOURCE]
    chosen.sort()  # preserves some locality but doesn't bias sampling

    log.info("  %s: parsing %d / %d eligible lines...",
             source, len(chosen), len(eligible_idxs))

    # Parse with nlp.pipe (respect batching for speed).
    texts_with_idx = [(i, lines[i]) for i in chosen]
    texts = [t for _, t in texts_with_idx]
    docs = nlp.pipe(texts, batch_size=64)

    pool: List[Candidate] = []
    n_fragments = 0
    for (line_idx, text), doc in zip(texts_with_idx, docs):
        verbs = classify_finite_verbs(doc)
        # Filter out fragments — sentences with no finite verb are useless
        # for null-subject annotation. They'd otherwise flood the
        # "predicted_overt" stratum and waste annotator effort.
        if not verbs:
            n_fragments += 1
            continue
        predicted_null = any(v["subject_status"] == "null" for v in verbs)
        pool.append(Candidate(
            source=source,
            line_idx=line_idx,
            text=text,
            predicted_null=predicted_null,
            predicted_verbs=verbs,
        ))
    if n_fragments:
        log.info("  %s: dropped %d fragments (no finite verb)",
                 source, n_fragments)
    return pool


# ── Sampling ─────────────────────────────────────────────────────────

def sample_iid(
    pool: List[Candidate],
    n: int,
    rng: random.Random,
) -> List[Candidate]:
    """Uniform random sample of size n from the pool."""
    if n >= len(pool):
        return list(pool)
    return rng.sample(pool, n)


def sample_stratified(
    pool: List[Candidate],
    n_per_class: int,
    rng: random.Random,
) -> List[Candidate]:
    """Sample n_per_class predicted-null + n_per_class predicted-overt.

    If one class is short, pad from the opposite class and log.
    """
    nulls = [c for c in pool if c.predicted_null]
    overts = [c for c in pool if not c.predicted_null]

    picked_nulls = rng.sample(nulls, min(n_per_class, len(nulls)))
    picked_overts = rng.sample(overts, min(n_per_class, len(overts)))

    short_nulls = n_per_class - len(picked_nulls)
    short_overts = n_per_class - len(picked_overts)
    if short_nulls > 0:
        remaining = [c for c in overts if c not in picked_overts]
        extra = rng.sample(remaining, min(short_nulls, len(remaining)))
        picked_overts.extend(extra)
        log.info("    stratified: %d predicted_null short, padded with overt",
                 short_nulls)
    if short_overts > 0:
        remaining = [c for c in nulls if c not in picked_nulls]
        extra = rng.sample(remaining, min(short_overts, len(remaining)))
        picked_nulls.extend(extra)
        log.info("    stratified: %d predicted_overt short, padded with null",
                 short_overts)

    return picked_nulls + picked_overts


# ── Context retrieval ────────────────────────────────────────────────

def build_context(
    lines: List[str],
    line_idx: int,
    n_before: int,
    n_after: int,
) -> Tuple[List[str], List[str]]:
    """Return (context_before, context_after) — each nearest-first.

    context_before: [line_idx-1, line_idx-2, ...]  (nearest immediately
    preceding, then further back).
    context_after:  [line_idx+1, line_idx+2, ...]  (nearest immediately
    following, then further ahead).

    Matches the ordering expected by
    ``annotation/db.py:_normalize_context``. Blanks are skipped.
    """
    before: List[str] = []
    i = line_idx - 1
    while i >= 0 and len(before) < n_before:
        if lines[i]:
            before.append(lines[i])
        i -= 1

    after: List[str] = []
    j = line_idx + 1
    while j < len(lines) and len(after) < n_after:
        if lines[j]:
            after.append(lines[j])
        j += 1

    return before, after


# ── Main ─────────────────────────────────────────────────────────────

def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Spanish null-subject gold annotation sample."
    )
    parser.add_argument(
        "--source_dir", type=Path, default=Path("data/spanish/train_90M"),
        help="Directory containing source .train files.",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("data/spanish/gold"),
        help="Output directory for stimuli.jsonl.",
    )
    parser.add_argument(
        "--only_source", type=str, default=None,
        help="If set, process only this source (for dry runs).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    # ── Load spaCy ───
    log.info("Loading es_core_news_lg...")
    import spacy
    nlp = spacy.load("es_core_news_lg")
    spacy_version = f"es_core_news_lg-{nlp.meta.get('version', 'unknown')}"
    log.info("  Loaded %s", spacy_version)

    # ── Build per-source pools ───
    train_files = sorted(args.source_dir.glob("*.train"))
    if args.only_source:
        train_files = [f for f in train_files if f.stem == args.only_source]
        if not train_files:
            raise SystemExit(f"No train file for source '{args.only_source}'")

    source_lines: Dict[str, List[str]] = {}
    pools: Dict[str, List[Candidate]] = {}

    for train_path in train_files:
        source = train_path.stem
        log.info("Source: %s", source)
        lines = load_source_lines(train_path)
        source_lines[source] = lines
        pools[source] = build_candidate_pool(source, lines, nlp, rng)
        if pools[source]:
            null_rate = sum(c.predicted_null for c in pools[source]) / len(pools[source])
            log.info("  pool: %d candidates, %.1f%% predicted_null",
                     len(pools[source]), null_rate * 100)

    # ── Sample IID and stratified per source ───
    # Dedupe by exact text across ALL sources (subtitles etc. often repeat
    # short utterances like "Sí.", "Gracias.").
    selected: Dict[str, Tuple[Candidate, List[str]]] = {}
    n_dropped_dup = 0

    def _add(cand: Candidate, sample_type: str) -> None:
        nonlocal n_dropped_dup
        key = cand.text
        if key in selected:
            existing_cand, types = selected[key]
            if sample_type not in types:
                types.append(sample_type)
            # Dropped the duplicate candidate itself (kept first encountered)
            if existing_cand is not cand:
                n_dropped_dup += 1
        else:
            selected[key] = (cand, [sample_type])

    for source, pool in pools.items():
        if not pool:
            continue
        log.info("Sampling from %s (pool=%d)...", source, len(pool))
        iid = sample_iid(pool, IID_PER_SOURCE, rng)
        strat = sample_stratified(pool, STRATIFIED_PER_CLASS_PER_SOURCE, rng)

        for c in iid:
            _add(c, "iid")
        for c in strat:
            _add(c, "stratified")

    log.info("Selected %d unique stimuli (dedup dropped %d)",
             len(selected), n_dropped_dup)

    # ── Sample IAA subset ───
    all_keys = list(selected.keys())
    rng.shuffle(all_keys)
    iaa_keys = set(all_keys[:IAA_TOTAL])
    for k in iaa_keys:
        selected[k][1].append("iaa")

    # ── Emit JSONL ───
    git_sha = _git_sha()
    out_path = args.output_dir / "stimuli.jsonl"
    n_written = 0
    n_iid = 0
    n_strat = 0
    n_iaa = 0

    # Sort for determinism: by source, then by line_idx.
    sorted_keys = sorted(
        selected.keys(),
        key=lambda k: (selected[k][0].source, selected[k][0].line_idx),
    )

    with open(out_path, "w", encoding="utf-8") as f:
        for key in sorted_keys:
            cand, sample_types = selected[key]
            before, after = build_context(
                source_lines[cand.source],
                cand.line_idx,
                CONTEXT_BEFORE_N,
                CONTEXT_AFTER_N,
            )
            record = {
                "text": cand.text,
                "source": cand.source,
                "context_before": before,
                "context_after": after,
                "metadata": {
                    "sample_types": sample_types,
                    "predicted_null": cand.predicted_null,
                    "predicted_verbs": cand.predicted_verbs,
                    "source_file": f"{cand.source}.train",
                    "source_line": cand.line_idx,
                    "spacy_model": spacy_version,
                    "sampler_commit": git_sha,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1
            if "iid" in sample_types:
                n_iid += 1
            if "stratified" in sample_types:
                n_strat += 1
            if "iaa" in sample_types:
                n_iaa += 1

    log.info("=" * 60)
    log.info("Wrote %d stimuli to %s", n_written, out_path)
    log.info("  IID:        %d", n_iid)
    log.info("  Stratified: %d", n_strat)
    log.info("  IAA:        %d", n_iaa)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
