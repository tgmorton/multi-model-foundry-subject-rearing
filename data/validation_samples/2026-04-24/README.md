# Ablation validation samples — 2026-04-24

Sampled from `/mnt/data/manipulations/{es,en}/<slug>/` on the `subject-drop-archive` PVC.

## Two formats per ablation

### `<slug>.tsv` — three-step sampler output

From `scripts/validate_ablation.py --mode three-step`. Schema:
```
genre  source  line_num  original  ablated  correct?  notes
```

`source` ∈ `train-kept`, `train-removed`, `pool-backfill`. Useful for
**line-removal ablations** (`remove_expletive_sentences`) — you can see
the expletive sentences that got dropped and the pool lines that filled
the deficit.

For **substitution ablations** (`impoverish_case`, `lemmatize_verbs`,
`enrich_verbal_morphology`) the sampler can't find substituted lines
verbatim in the composed file, so the `ablated` column shows
`<REMOVED>`. Use the before/after files instead.

### `before_after_<slug>_<genre>.tsv` — substitution-aware sampling

Random sample of lines from the raw vs composed corpus where they
differ. Schema:
```
line_num  original  ablated
```

40 examples per (slug, genre). Genres covered: childes (both langs),
europarl (es), bnc_spoken (en).

## Known issue (EN lemmatize_verbs, EN enrich_verbal_morphology)

Both still reflect the **contraction-glue bug** from the overnight first
pass. Look for pseudo-tokens like `itbe`, `webe`, `ben't`, `doon't`,
`whatbe` in the EN substitution samples — these are surface artefacts
of contractions getting glued to their replacements without
re-inserting whitespace.

Fix is committed (`084c515`) but the re-run is blocked on the NRP
admission webhook quota. Once NRP clears, re-dispatching
`thomas-ablate-compose-en-v1` overwrites the buggy outputs and a
fresh sample will be clean.

The other 5 ablations (all 3 ES + EN remove_expletive_sentences + EN
impoverish_case) are clean.
