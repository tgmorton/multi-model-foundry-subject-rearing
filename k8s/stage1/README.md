# English Stage-1 early-seed manifests

These manifests implement the informally preregistered Stage-1 design:

- 6 architectures × the existing 5 English conditions = 30 cells.
- The same 12 seeds are used in every cell.
- Seeds `42` and `137` are existing full-run anchors and are not relaunched.
- Only selected hyperparameter rank `h0` is used.
- New trajectories stop after two epochs.
- The all-architecture wave keeps `parallelism: 1`. Separate validated tranche
  files are provided for GPT-2-small, GPT-2-medium, and LSTM at
  `parallelism: 2` for the staged rollout.
- Training preserves the legacy English-wave code/image tuple so new early
  trajectories remain comparable to the existing anchors.

The common seed set is:

```text
42
137
314159
1568568120
1415936399
1640623142
1595274352
2022192329
1891911437
1881877998
302485963
2078344582
```

## Model-appropriate h0 settings

The generator imports `ARCH_SETTINGS` from the production launcher and lets
`production_agent.py` resolve the selected rank-0 winner from
`data/sweep_winners/{arch}_en.json`. The resulting training tuples are:

| Architecture | Physical batch | Effective batch | Accumulation | Pod RAM | Pod CPU |
|---|---:|---:|---:|---:|---:|
| GPT-2 small | 16 | 128 | 8 | 4 GiB | 2 |
| GPT-2 medium | 16 | 128 | 8 | 4 GiB | 2 |
| GPT-2 large | 4 | 512 | 128 | 5 GiB | 2 |
| BERT large | 4 | 256 | 64 | 8 GiB | 2 |
| LSTM | 16 | 128 | 8 | 4 GiB | 2 |
| Mamba 370M | 4 | 512 | 128 | 5 GiB | 2 |

All use sequence length 1,000, one GPU, and the exact four-product 24 GB pool
from the production launcher: RTX 3090, A10, L4, and RTX 4090.

## Files and sequencing

1. `sentinels/` contains five separate countable baseline sentinel YAMLs, one
   for each architecture not already validated by the completed GPT-2-small
   seed-314159 sentinel. Submit and review them **one at a time**; the files
   are deliberately separate to prevent an accidental sentinel fleet.
2. `job-stage1-early-wave-en.yaml`
   contains one serial Indexed Job for every cell. Baseline seed 314159 is
   omitted because it is completed/planned in the sentinel phase.
3. `job-stage1-early-wave-{gpt2-small,gpt2-medium,lstm}-p2-en.yaml` contains
   five Jobs for one validated architecture, with `parallelism: 2`. These
   tranche files must not be submitted together with the corresponding Jobs
   in the all-architecture wave. Each indexed seed has a high emergency retry
   ceiling of 100 so transient cluster failures do not drop a seed. Success
   advances immediately; the seven-day deadline and recurring health monitor
   remain the operational guards.

The wave file contains 30 Jobs and 294 trajectories. The sentinel file adds
five trajectories. Together with the completed GPT-2-small baseline sentinel,
the manifests cover the 300 new trajectories required after excluding the
60 existing seed-42/137 anchors.

## Generation and validation

Regenerate deterministically from the production launcher:

```bash
python scripts/generate_stage1_early_manifests.py
```

Before any submission:

1. Verify the historical seed-42/137 runs are provenance-equivalent in all
   30 cells.
2. Verify every proposed run ID is absent from S3, WandB, Kubernetes, and the
   checkpoint PVC.
3. Server-dry-run both files against explicit context `nautilus` and namespace
   `lemn-lab`.
4. Submit one architecture sentinel.
5. Confirm its GPU/VRAM, CPU/RAM, I/O, checkpoint, registry, and evaluation
   telemetry before advancing to the next sentinel.
6. Consider the 30-cell wave only after all five architecture tuples pass.

The YAMLs are design artifacts only. Their presence does not authorize or
perform a Kubernetes submission.
