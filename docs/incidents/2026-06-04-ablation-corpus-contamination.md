# Incident: ablation corpus contamination via recursive glob

**Date discovered:** 2026-06-04
**Severity:** scientific — invalidated baseline-vs-ablation contrasts for all trained ablation models
**Status:** remediated (clean re-run); remediation code 2026-06-05

---

## Summary

A recursive glob `**/*.train` in `tokenize_dataset.py` ingested the compose
**intermediate** directories for every ablation corpus alongside the final
composed `.train` file. The intermediates include `_train/` (an approximately
full duplicate of the composed corpus), `_pool/`, and `pool_remainder/`. As a
result, every ablation training cache was about **2.2× the size of the baseline**
(281,765 vs 128,069 chunks of 1000 tokens). Each ablation sentence was seen
roughly **2×/epoch**, plus the extra 10M-word pool. The baseline corpus was built
by a path that produced no such intermediates and was **clean**.

---

## Impact

- **Trained ablation models** — both the old training wave and the recovery
  resumes — are **scientifically invalid for baseline-vs-ablation contrasts**:
  ablation models saw their data roughly twice per epoch while the baseline saw
  it once, so any baseline-vs-ablation difference is confounded with exposure.
- **Phenomenon integrity is intact.** The ablated text itself (the linguistic
  manipulation) was correct; only the *quantity* of exposure was wrong. The
  contamination was duplication of the ablated corpus, not injection of
  non-ablated content.

---

## Detection

The PI noticed on the fleet dashboard (2026-06-04) that **baseline epochs were
about 2.2× shorter** than ablation epochs. Equal-word-count conditions should
have near-equal epoch lengths; the baseline being the short one pointed straight
at extra data in the ablation caches.

---

## Root causes

1. **Ingestion trusted the directory layout.** `tokenize_dataset.py` globbed
   `**/*.train` recursively and assumed every `.train` under a manipulation
   directory was production input. The compose intermediates (`_train/`,
   `_pool/`, `pool_remainder/`) also carry `.train` files and were swept in.
2. **No conservation assertion.** Nothing checked the ingested word/chunk count
   against the corpus manifest, so a 2.2× blow-up passed silently.
3. **Cache keys hashed the path, not the content.** The content-addressed cache
   keyed on the corpus path string, so a corpus with the wrong content but the
   expected path produced a "valid" cache hit and a recorded `cache_key` that did
   not fingerprint what was actually ingested.
4. **A prior warning was misread.** An earlier `approx_chunks=127000` estimate vs
   the real **281,765** chunks was dismissed as a stale estimate rather than
   investigated. The estimate was in fact correct — it matched the **clean**
   corpus (~128,069 chunks) — and the discrepancy was the contamination
   announcing itself.

---

## Remediations

- **Manifest-driven ingestion** with checksum verification: ingestion reads the
  corpus manifest and verifies checksums rather than discovering files by glob.
- **Top-level-only globs:** file discovery no longer recurses into compose
  intermediate subdirectories.
- **Content-hashed cache keys:** the cache key binds the corpus *content*, so a
  contaminated corpus cannot collide with a clean one at the same path.
- **Full clean re-run.** Because clean ablation corpora are ~2.2× smaller, the
  clean runs are ~2.2× shorter, roughly cancelling the cost of re-running.

These remediations are recorded as decisions #1 and #5 in
[docs/design_ledger.md](../design_ledger.md).

---

## Timeline

| When | Event |
|---|---|
| ~May 13 | Contaminated ablation caches built. |
| May 13–28 | Old training wave runs on contaminated ablation caches. |
| Jun 4 | PI notices baseline epochs ~2.2× shorter on the fleet dashboard; contamination diagnosed. |
| Jun 5 | Remediation code landed (manifest-driven ingestion + checksum verify, top-level-only globs, content-hashed cache keys); clean re-run initiated. |
