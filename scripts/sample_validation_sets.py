"""Fix the validation coding sheets:

1. For line-removal ablations (remove_expletive_sentences): use proper
   kept/removed labelling (a raw line is 'kept' if it appears verbatim in
   the composed file, else 'removed' with ablated='<REMOVED>'). The
   previous custom sampler paired raw[i] vs composed[i] which is wrong
   because composed lines are reshuffled.

2. For ALL CSVs: write with UTF-8 BOM (encoding='utf-8-sig'). Excel and
   many other CSV viewers default to Latin-1 / Windows-1252 without the
   BOM and render Spanish accents as mojibake.

Runs inside thomas-pod-archive (has subject-drop-archive PVC mounted).
"""
import csv, json, os, random
import sys

OUT = "/tmp/fixed_coding_sheets"
os.makedirs(OUT, exist_ok=True)
N_PER_GENRE = 50  # for line-removal, stratified across kept + removed + backfill
N_INSPECT = 250
random.seed(42)

# Configurations: per (lang, slug), what's the genre list and is it line-removal?
JOBS = [
    ("es", "remove_expletive_sentences", True,
     ["childes", "europarl", "opensubtitles", "vikidia", "qed", "child_narratives"]),
    ("es", "impoverish_case", False,
     ["childes", "europarl", "opensubtitles", "vikidia", "qed", "child_narratives"]),
    ("es", "lemmatize_verbs", False,
     ["childes", "europarl", "opensubtitles", "vikidia", "qed", "child_narratives"]),
    ("en", "remove_expletive_sentences", True,
     ["bnc_spoken", "childes", "gutenberg", "open_subtitles", "simple_wiki", "switchboard"]),
    # The 3 EN substitution ablations (impoverish_case, lemmatize_verbs,
    # enrich_verbal_morphology) are re-sampled separately AFTER re-run #3
    # completes — they need the contraction fix + past tense + suppletion
    # to be reflected. Re-sampling those here would use stale data.
    ("en", "impoverish_case", False,
     ["bnc_spoken", "childes", "gutenberg", "open_subtitles", "simple_wiki", "switchboard"]),
]


def sample_line_removal(lang, slug, genres):
    """Three-way stratified sample: train-kept, train-removed, pool-backfill.
    Returns list of dicts with keys: source, genre, line_num, original, ablated."""
    rows = []
    for genre in genres:
        random.seed(42 + hash(genre) % (1 << 20))
        raw_path = f"/mnt/data/raw/{lang}/train_90M/{genre}.train"
        composed_path = f"/mnt/data/manipulations/{lang}/{slug}/{genre}.train"
        pool_path = f"/mnt/data/manipulations/{lang}/{slug}/_pool/{genre}.train"
        if not (os.path.exists(raw_path) and os.path.exists(composed_path)):
            print(f"  skip {lang}/{slug}/{genre}", flush=True)
            continue
        with open(raw_path) as f:
            raw_lines = f.readlines()
        with open(composed_path) as f:
            composed_set = set(line.rstrip("\n") for line in f if line.strip())

        # Categorize raw lines
        kept_idx, removed_idx = [], []
        for i, line in enumerate(raw_lines):
            stripped = line.rstrip("\n")
            if not stripped.strip():
                continue
            if stripped in composed_set:
                kept_idx.append(i)
            else:
                removed_idx.append(i)

        # Sample 1/3 kept, 1/3 removed, 1/3 pool-backfill per genre.
        n_each = max(1, N_PER_GENRE // 3)
        if kept_idx:
            for i in random.sample(kept_idx, min(n_each, len(kept_idx))):
                rows.append({"source": "train-kept", "genre": genre,
                             "line_num": i, "original": raw_lines[i].rstrip("\n"),
                             "ablated": raw_lines[i].rstrip("\n")})
        if removed_idx:
            for i in random.sample(removed_idx, min(n_each, len(removed_idx))):
                rows.append({"source": "train-removed", "genre": genre,
                             "line_num": i, "original": raw_lines[i].rstrip("\n"),
                             "ablated": "<REMOVED>"})
        # Pool-backfill: read the ablated pool file directly, sample lines
        if os.path.exists(pool_path):
            with open(pool_path) as f:
                pool_lines = [l.rstrip("\n") for l in f if l.strip()]
            if pool_lines:
                for i in random.sample(range(len(pool_lines)),
                                       min(n_each, len(pool_lines))):
                    rows.append({"source": "pool-backfill", "genre": genre,
                                 "line_num": i, "original": "<pool sample>",
                                 "ablated": pool_lines[i]})
    return rows


def sample_substitution(lang, slug, genres):
    """For substitution ablations: pair raw[i] with composed[i] for lines
    that actually differ. Line counts are preserved for substitution
    ablations so the index-pairing is meaningful here."""
    rows = []
    for genre in genres:
        random.seed(42 + hash(genre) % (1 << 20))
        raw_path = f"/mnt/data/raw/{lang}/train_90M/{genre}.train"
        abl_path = f"/mnt/data/manipulations/{lang}/{slug}/{genre}.train"
        if not (os.path.exists(raw_path) and os.path.exists(abl_path)):
            continue
        with open(raw_path) as f:
            raw_lines = f.readlines()
        with open(abl_path) as f:
            abl_lines = f.readlines()
        n = min(len(raw_lines), len(abl_lines))
        changed = [i for i in range(n)
                   if raw_lines[i] != abl_lines[i] and raw_lines[i].strip() and abl_lines[i].strip()]
        if not changed:
            continue
        idx = random.sample(changed, min(N_PER_GENRE, len(changed)))
        for i in sorted(idx):
            rows.append({"source": "train-modified", "genre": genre,
                         "line_num": i, "original": raw_lines[i].rstrip("\n"),
                         "ablated": abl_lines[i].rstrip("\n")})
    return rows


for lang, slug, is_removal, genres in JOBS:
    if is_removal:
        rows = sample_line_removal(lang, slug, genres)
    else:
        rows = sample_substitution(lang, slug, genres)
    if not rows:
        print(f"  {lang}/{slug}: 0 rows (skipped)")
        continue
    # Truncate to N_INSPECT
    rows = rows[:N_INSPECT]
    out_path = f"{OUT}/coding_sheet_{lang}_{slug}.csv"
    # UTF-8 BOM (utf-8-sig) so Excel and other apps correctly decode Spanish accents.
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "source", "genre", "line_num", "original",
                    "ablated", "verdict", "category_hit", "notes"])
        for i, r in enumerate(rows):
            w.writerow([i, r["source"], r["genre"], r["line_num"],
                        r["original"], r["ablated"], "", "", ""])
    print(f"  {lang}/{slug}: {len(rows)} rows written to {out_path}")
