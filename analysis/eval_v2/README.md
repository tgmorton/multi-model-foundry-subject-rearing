# eval_v2 analysis

Experimental notebooks + figures for the null-subject eval waves.
Data comes from S3 (`eval_results/null_subj_v2/`), pulled locally via:

```bash
AWS_PROFILE=nrp python scripts/pull_eval_results.py            # items+pairs+checkpoints
AWS_PROFILE=nrp python scripts/pull_eval_results.py --tables per_token
```

Conventions:
- one dated notebook per exploration (`notebooks/YYYY-MM-DD_*.ipynb`);
  notebooks are reproducible from a fresh pull
- figures land in `figures/`, named `<scope>_<what>.png`
- wave-level caveats (stimulus bugs, scoring versions) recorded in the
  notebook header so plots aren't reused without their asterisks

Status dashboard: `evalfleet` (alias for `scripts/eval_status.py`).
