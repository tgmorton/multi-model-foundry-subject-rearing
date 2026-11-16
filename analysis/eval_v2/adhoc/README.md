# One-off analyses behind numbers sent to the talk deck

Scripts here were written for a single question and are kept so the numbers can be reproduced.

| Script | Question | Where the numbers went |
|---|---|---|
| `compare_enrich_rescore.py` | Do the enrichment re-scores (matched v2) agree with the August scores on unchanged sentences? | Enrichment verification for gpt2_small, gpt2_medium, gpt2_large, lstm, mamba_370m |
| `priming_components.py` | In priming, does the overt prime lower the overt target or raise the null target? | Negative-priming answer (he / she competition), 2026-11-13 |
| `that_verb_counts.py` | Does pronoun removal change the count of "that" + verb strings in the training text? | That-trace string counts, 2026-11-16 |

## Numbers sent to the deck that have no script yet

These came from inline analyses in the 2026-11-13/16 session and would need to be re-derived:

- Early dip below chance: by-pronoun preference and the pronoun / sentence-average / following-words
  decomposition, on the decile trajectories (`null_subj_test10m_robbi_nat_traj_v1`, per_token) and on the
  matched evaluation (12 unmodified gpt2_large seeds, per_token pulled from S3).
- Token-learning front: the model's average log-probability per token by corpus log-frequency band
  (`data/unigrams/en_baseline__shared_unigram.pkl`).
- Pronoun-free contexts against original contexts in early training (report cache `priming_items`, unprimed rows).
- Priming re-cut: effect by relation, the sentence-initial block, object- against subject-position primes,
  effect by removal order (from `report_cache/priming_deep_fixed/{end,meta}.parquet`).
- Clustering of removed pronouns and of repeated he / she in the training text (`survey/robbi_joined.parquet`).
- That-trace: object- and subject-gap preference by architecture, condition and matrix verb; correlation of
  subject-gap preference with string counts across the 27 partial-removal models.
- Training counts from the run registry (`run_registry/by_run/*.json`; `registry.parquet` is stale since 2026-09-28).
