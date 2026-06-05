# Design Ledger

Final design decisions for the subject-drop study, decided by the PI on
2026-06-04/05. This ledger records what was adopted, what was rejected, and the
reasoning, so the decisions are not re-litigated. It consolidates the dispositions
of the two audits:

- [docs/audits/optimization_audit_2026-06-04.md](audits/optimization_audit_2026-06-04.md)
- [docs/audits/replicability_audit_2026-06-04.md](audits/replicability_audit_2026-06-04.md)

Related incident: [docs/incidents/2026-06-04-ablation-corpus-contamination.md](incidents/2026-06-04-ablation-corpus-contamination.md)

---

## Numbered decisions

### #1 — Identity-pipeline baseline — ADOPTED

The baseline trains on `data/manipulations/{lang}/baseline/`, built by the **same
ablation → compose pipeline** as every ablation but with an **identity
transformation**. The only difference between the baseline corpus and an ablation
corpus is the manipulation itself; ingestion, composition, tokenization, and
chunking are byte-for-byte the same code path. This removes the prior asymmetry
(baseline built by a different path than ablations) that the contamination
incident exposed.

### #2 — Exposure matching — WORD-MATCHED

Confirmed as the existing behavior: compose matches **whitespace tokens**
(~92.07M words per condition). Word-matching is the design choice.

- Per-condition **subword token counts** will be reported (they differ slightly
  by condition because manipulations change the text).
- **Token-matching is explicitly rejected** as the matching target.

### #3 — Paired initialization — ADOPTED (now structural)

Same `(arch, seed)` produces **identical init weights across all conditions**.
This was previously true only by accident (nothing consumed torch RNG between
`set_seed` and model construction). It is now made structural:

- `set_seed()` is re-called **immediately before model construction**.
- An assertion test builds the same `(arch, seed)` under two conditions and
  asserts bitwise-equal initial `state_dict`.
- A checkpoint-0 cross-condition audit is available to verify paired init in
  flight.

### #4 — HP allocation — KEPT (5 HP ranks × 2 seeds per cell)

5 HP ranks × 2 seeds per cell is retained. Advisor preference is for **HP
diversity over seed replication**.

### #5 — Closed corpus artifact — ADOPTED

The corpus is treated as a closed, manifest-described artifact:

- **Manifest-driven ingestion** with checksum verification.
- **Top-level-only** file discovery (no recursive globbing into compose
  intermediates — see the contamination incident).
- **Content-hashed cache keys** (the cache key binds corpus content, not path).
- The **registry records the corpus manifest checksum** per run.

### #6 — Spanish condition count — 3 conditions

Spanish stays at **3 conditions**; there is **no ES `enrich_verbal_morphology`**.
The preregistration is advisory on this point.

---

## Seeds and data order

- Seeds **{42, 137}** are kept in **every cell**.
- **Within-arch pairing is the load-bearing structure** of the design.
- **Per-arch seed labels** were considered and **declined** — judged
  statistically neutral.
- **Data order** is a designed **pure function of `(seed, epoch)`** via a
  dedicated generator. It is **NOT arch-salted**: at equal seed, every arch sees
  the same data order.
- **Open option (noted, not committed):** an optional calibration cell
  (1 arch × ~10 seeds) to estimate σ_seed.

---

## Sizing nomenclature

The configs are correct; the prior labels were misleading. Actual parameter
counts:

| Internal arch id | Actual params | Shape note |
|---|---|---|
| `gpt2_small` | 45M | |
| `gpt2_medium` | 124M | ≡ GPT-2-small shape |
| `gpt2_large` | 355M | ≡ GPT-2-medium shape |
| `bert_large` | 355M | |
| `lstm` | 30M | |
| `mamba_370m` | 371M | |

- The **paper reports literal sizes** (GPT-2-45M / 124M / 355M).
- **Internal arch ids are unchanged** (no renames).
- **`n_params` is recorded per run in the registry.**
- Note the felicitous **size match at the top tier**: gpt2_large 355M ≈
  bert_large 355M ≈ mamba_370m 371M.

---

## Determinism

- **Production claims statistical replication.** FA2 and Mamba kernels use
  `atomicAdd`-based nondeterministic reductions, so production runs are not
  bitwise reproducible even on a fixed GPU.
- A **`training.deterministic` verification subset** (SDPA + `use_deterministic_algorithms`
  + TF32-off) demonstrates **bitwise same-GPU reproducibility where the kernels
  allow** (e.g. LSTM).
- **All controllable RNG streams are fixed:** weight init, data order, MLM masks,
  and dropout.

---

## Performance adoptions (from the optimization audit)

| Lever | Disposition | Rationale |
|---|---|---|
| Gradient checkpointing off — GPT-2 family | **Adopted** | Largest safe lever; ~25% per-step, math-identical. (Mamba/BERT GC is a dead flag — removed for honesty, 0 gain.) |
| `torch.compile` `"default"` for transformers | **Adopted, A/B parity-gated** | Commit only after a 1-seed loss-curve parity check vs eager. |
| fp16 | **Kept** (bf16 declined) | Sweep winners were swept under fp16; switching dtype would invalidate them. |
| A100-80GB routing | **Declined** | Width-over-speed: prefer breadth across the 100+ open-pool GPUs over reservation-gated speed. |
| L4 stratification | **Deferred** | Not adopted now; revisit if the L4 tail proves costly. |
| int32 chunk storage | **Adopted** | Chunk token storage uses int32. |
| Node-local cache copy | **Deferred** | Low value. |
| Async checkpoint writes | **Deferred** | Low value. |
