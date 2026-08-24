# 95-Cell Matrix Compose — Verification Record

2026-08-23, post-compose audit of `thomas-ablate-compose-matrix-v1`
(95/95 completions, 139 min). Verifier job `thomas-matrix-verify-v1`
read every cell's `ABLATION_MANIFEST.json` / `COMPOSE_MANIFEST.json` on
`/mnt/data/manipulations/en/` and emitted per-cell counts
(`matrix_actual.json`); expectations (`matrix_expected.json`) were
derived locally from the frozen selection v4 parquets
(`data/recoverability/analysis/selection/{train_90M,pull_10M}/`,
scorer `bert_wwm_r1`, 6,457,919 instances).

## Result

- **95/95 cells present, 0 errors.**
- **76/76 non-expletive cells** (`base`, `impcase`, `lemverb`,
  `enrichvm` × 19 (arm, k)): `train_removed` and `pool_removed` match
  the v4 selection **exactly**, `exhausted == 0`,
  `short_after_pool == 0`. Interventions rewrite tokens but never
  change which pronouns are removed — the same-pronouns contract holds.
- **19/19 expletive cells** (`rmexpl`): composed under `--allow-short`
  per precedent (the pre-matrix expletive corpus already ran
  −3,127,412 words on gutenberg). Removal counts differ from the raw
  selection because expletive **line** removals subsume the selected
  pronouns on those lines (not separately counted) while pool backfill
  brings its own pool-table pronoun removals. Recorded deficits
  (words short after pool exhaustion):

| cell (info / rand) | deficit info | deficit rand | exhausted genres |
|---|--:|--:|--:|
| k=10 | 3,355,692 | 3,351,570 | 2 / 2 |
| k=20 | 3,557,040 | 3,518,757 | 2 / 2 |
| k=30 | 3,771,498 | 3,686,220 | 2 / 2 |
| k=40 | 3,986,281 | 3,853,372 | 2 / 2 |
| k=50 | 4,188,486 | 4,020,811 | 3 / 2 |
| k=60 | 4,366,705 | 4,188,210 | 3 / 2 |
| k=70 | 4,512,851 | 4,361,431 | 3 / 3 |
| k=80 | 4,638,572 | 4,538,965 | 3 / 3 |
| k=90 | 4,751,870 | 4,716,093 | 3 / 3 |
| k=100 (shared) | 4,904,056 | — | 4 |

  Deficit grows monotonically with k (~3.35M → 4.90M words, i.e.
  cells land at ~85.1–86.7M words instead of 90M): higher k removes
  more pronoun-bearing words, demanding more backfill from
  already-exhausted genre pools (gutenberg-dominated). **Note for
  analysis**: within the rmexpl sweep, corpus size co-varies with k by
  up to ~1.5M words — a mild confound to keep in mind (or to eliminate
  via the pre-authorized Gutenberg pool expansion, which would require
  annotate → BERT-score → pool-table extension → recompose of these
  19 cells).
- **PVC**: 49.0 TB free after compose.

## Files

- `matrix_expected.json` — expected removals per `corpus:arm:k` from
  selection v4.
- `matrix_actual.json` — per-cell
  `{slug, train_removed, pool_removed, exhausted, short_after_pool, over_target}`.
- S3 mirror: `s3://thomas-subject-drop-artifacts/recoverability/analysis/matrix_verification/`.
