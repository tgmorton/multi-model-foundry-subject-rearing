# HP sweep configs

This directory has two kinds of YAML:

- **Sweep configs** (`<arch>_<lang>.yaml`) — WandB sweep specs with
  `method: bayes`, Hyperband early-stop, and the HP search space. One
  per (arch, lang) pair. Fed to `wandb sweep` to register a sweep and
  get a sweep ID.
- **Baselines** (`baselines/<arch>_<lang>.yaml`) — the frozen-architecture
  starting point. `scripts/sweep_agent_lm.py` reads this, overlays the
  sampled HPs from `wandb.config`, trains for the proxy horizon (3
  epochs), logs `proxy/held_out_perplexity` + `proxy/final_training_loss`.

## Operator workflow

Prerequisites (one-time per (arch, lang) pair — assumes the
`subject-drop-archive` PVC is mounted):

1. Train the tokenizer. Edit `k8s/job-train-tokenizer.yaml` so
   `TOKENIZER_CONFIG` points at `configs/sweeps/baselines/<arch>_<lang>.yaml`
   (or any config with matching `tokenizer.output_dir`), then:

   ```bash
   kubectl apply -f k8s/job-train-tokenizer.yaml
   ```

   Wait for it to print `Tokenizer already exists at …` (idempotent; safe
   to reapply).

2. Tokenize the train + test corpora. Not currently a standalone K8s
   template — happens automatically when the sweep pod runs the prep
   block. If you want to decouple, launch any `tokenize-dataset`-only
   pod with the same config.

### Running a sweep

From your laptop (needs `wandb` CLI + `WANDB_API_KEY` in env or
`~/.netrc`):

```bash
wandb sweep --project subject-drop-sweeps \
            configs/sweeps/gpt2_medium_en.yaml
# → thmorton/subject-drop-sweeps/<SWEEP_ID>
```

Copy the sweep ID, then launch N agents on the cluster:

```bash
# Edit k8s/job-sweep-<arch>-<lang>.yaml: set SWEEP_ID to the value above.
# TRIAL_COUNT controls how many trials each pod runs sequentially.
kubectl apply -f k8s/job-sweep-gpt2-medium-en.yaml

# For more parallelism: apply copies of the same job with different
# metadata.name — each gets scheduled independently.
```

Monitor in WandB UI: `https://wandb.ai/<your-entity>/subject-drop-sweeps`
under the sweep ID.

### After convergence

```bash
# Pick the winner (rank trials by held_out_perplexity, mark rank=1 as
# is_hp_winner=True in the registry)
python scripts/select_hp_winner.py --arch gpt2_medium --lang en
```

The winner's `hyperparameters` dict becomes the frozen HP vector for
production training of that (arch, lang) — read it back from the
registry:

```python
from model_foundry.registry import iter_all_records
winner = next(r for r in iter_all_records()
              if r.get("arch") == "gpt2_medium"
              and r.get("lang") == "en"
              and r.get("is_hp_winner"))
print(winner["hyperparameters"])
```

## The 12 sweeps

Per the "separate sweeps per language" decision, 6 archs × 2 langs = 12
sweeps total. Each has its own `<arch>_<lang>.yaml` + `baselines/<arch>_<lang>.yaml`:

| Arch | en | es |
|---|---|---|
| gpt2_small | todo | todo |
| gpt2_medium | **done** | todo |
| gpt2_large | todo | todo |
| bert_large | todo | todo |
| lstm | todo | todo |
| mamba_370m | todo | todo |

(n-gram models don't need HP sweeps.)

Adding a new (arch, lang) sweep is mostly copy+sed: duplicate both
YAMLs, change the `arch`/`lang`/`base_config` constants, adjust the
model block in the baseline to match the architecture's identity
params. The HP search space is the same for every transformer/LSTM/Mamba.

## Notes on the metric choice

The sweep YAML ranks on `proxy/held_out_perplexity`. We also log
`proxy/final_training_loss` for every trial. Per the locked decision
("run A and B and decide from there"), we'll cross-check rankings
after the first sweep converges. If the two metrics agree on the
winner, we continue with perplexity-only ranking for the remaining 11
sweeps. If they disagree meaningfully, revisit.
