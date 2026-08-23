# BERT Context-Locality Landscape for Subject-Pronoun Recoverability

2026-08-24. Directive: map how much context — symmetric, backward-only,
forward-only, asymmetric — masked recovery of subject pronouns needs;
find where it saturates; test the "speaker model" hypothesis (deep
backward + limited-but-nonzero forward context).

**Design.** bert-large-uncased-wwm; frozen 100K-instance sample
(genre-proportional, versioned in git); 50 distinct (L, R) stream-context
configs — symmetric 0–250 wordpieces, backward-only to 500, forward-only
to 250, asymmetric cross L∈{16,32,64,250} × R∈{1..64}; identical
instances under every config (paired). Full metrics:
`data/recoverability/analysis/locality/config_metrics.csv`; figures under
`.../locality/figures/`.

## Findings

1. **Forward context is the saturation driver.** At any backward depth,
   the first forward wordpieces collapse the measure: at L=250, median
   surprisal falls 6.72 → 2.68 → 0.54 → 0.11 nats as R goes 0 → 1 → 2 → 8.
   By R≈8 the measure is deeply saturated regardless of L (tie mass
   ≥ 48%). The upcoming verb is most of what makes a subject pronoun
   "recoverable" to a bidirectional reader.
2. **Backward-only never saturates.** Even at L=500/R=0: median 5.83
   nats, tie mass 1.1% — discriminative at every depth tested, with the
   highest causal-ensemble agreement in the BERT family (ρ rises 0.35 →
   0.50 → 0.54 from L=1 to L=500, still climbing at the budget edge).
3. **Forward-only is a different signal entirely**: its ranking is
   slightly NEGATIVELY correlated with speaker predictability
   (ρ ≈ −0.05 to −0.08 for R ≥ 4) while converging fast toward the
   full-bidirectional ceiling. Listener-side (verb-agreement-driven)
   recoverability and speaker-side predictability rank the corpus in
   nearly unrelated orders — a clean dissociation worth reporting on its
   own.
4. **The speaker-model region behaves exactly as hypothesized.**
   L≫R with R=1: de-saturated (median ≈ 2.6–2.7 nats, tie mass ≈ 10%),
   best in-family causal agreement (ρ = 0.513 at 64:1, **0.525 at
   250:1**), and conceptually "a speaker one word ahead." R=2 is the
   knife edge (median ≈ 0.55, ties ≈ 24%); R ≥ 4 tips into saturation.
5. **Person decomposition (register effects).** Backward-only: 2nd
   person gains most from deep history (9.59 → 5.03 nats by L=500 —
   dialogue "you" lives in the turn structure); 3rd person stays hardest
   at every depth (6.13 at L=500 — antecedent access); 1st intermediate.
   In the speaker region (R=1), 2nd person is easiest (≈1.8), 1st
   hardest (≈3.2) — BERT's book/wiki register under-predicts speech-like
   "I" exactly as the earlier register analyses suggested.

## Rater candidates on the table

| Candidate | median (nats) | tie mass <0.1 | ρ vs causal ensemble | Construal |
|---|--:|--:|--:|---|
| gpt2m-external (frozen v2) | ~2.4 | ~0% | **0.775** | speaker (causal, in-register-adjacent) |
| BERT **250:1** | 2.68 | 9.9% | 0.525 | speaker one-word-ahead |
| BERT 500:0 | 5.83 | 1.1% | 0.540 | pure speaker history |
| BERT 250:2 | 0.54 | 24% | 0.466 | knife edge |
| BERT ±250 (bidirectional) | 0.06 | 57% | 0.26–0.33 | listener — SATURATED, unusable for ranking |

**Recommendation**: if the rater stays BERT, **L=250 R=1** is the
empirically supported speaker-model config — de-saturated,
discriminative, best in-family causal agreement, and matches the stated
hypothesis. gpt2m-external remains the strongest pure-agreement choice.
A defensible paper framing uses BERT-250:1 as the selection rater with
gpt2m + the in-house ensemble as banked convergent measures — plus
finding #3 (speaker/listener rank dissociation) and the bidirectional
saturation result as standalone empirical contributions.

Decision needed before the full-corpus rescore (one flag, ~2 h of GPU):
ranking rater = BERT-250:1, or gpt2m-external.
