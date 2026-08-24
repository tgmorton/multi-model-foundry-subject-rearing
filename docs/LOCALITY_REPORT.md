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

## Addendum (2026-08-24): causal-vs-BERT backward depth — why BERT "never saturates"

Thomas's hypothesis: BERT's flat backward curve is a *training-regime*
artifact — its pretraining always had the forward view (the verb), so it
never learned to squeeze backward cues; a causal LM, trained on the
backward view alone, should derive more from added backward depth.

**Design.** Pretrained gpt2-medium scored the identical frozen 100K
sample at the identical backward depths L ∈ {1,2,4,8,16,32,64,125,250,
500} (R structurally 0 under causal attention). Paired per-instance;
999,940 rows. Code: `--causal-ctx-grid` in
`scripts/score_pronoun_recoverability.py`; analysis:
`analysis/recoverability/causal_backward_analysis.py`; metrics:
`data/recoverability/analysis/locality/causal_backward_metrics.csv`.

**Verdict: the hypothesis is strongly supported.**

1. **The causal model extracts 2.2× more from backward context**:
   median gain L=1→500 is **3.32 nats (gpt2m) vs 1.54 (BERT)**, despite
   BERT being the larger model trained on more data.
2. **It extracts it locally and efficiently**: gpt2m captures 66% of
   its own total backward gain by L=16 and 91% by L=64, saturating
   smoothly — the classic incremental-prediction profile. BERT is the
   inverse: **76% of its total backward gain arrives after L=16**, and
   fully half of it in the last doubling (250→500).
3. **BERT's short-range backward use is qualitatively broken**, not
   just weaker: its curve is non-monotonic at small L (L=4 is *worse*
   than L=2 by 0.44 nats — truncated left-only fragments are
   out-of-distribution for an always-bidirectional reader).
4. **Reinterpretation of finding #2 above**: "backward-only never
   saturates" is not evidence that deep history holds unique
   information — gpt2m shows ~80% of the extractable information lives
   within 16 pieces. It's a symptom of BERT extracting backward
   information *inefficiently*, dribbling it in over ever-more
   redundant context. Exactly the "leans less on backward cues" story.
5. **Rank behavior**: gpt2m's ranking sharpens fast with depth
   (ρ vs the in-house causal ensemble 0.334 → **0.761** at L=500 —
   which recovers the frozen v2 external-gpt2m agreement of ~0.775,
   validating the pipeline end-to-end). BERT climbs only 0.355 → 0.540.
   Cross-model rank agreement at matched depth is moderate (0.41–0.54):
   the two readers extract partially different things from the same
   history.
6. **Person decomposition transfers**: both models find 2nd person
   easiest from pure history (dialogue turn structure) — the effect is
   architectural-regime-independent. But 3rd person is BERT's hardest
   at every depth while being on par with 1st for gpt2m: antecedent
   access from history is a causal-reader strength.

**Implication for the rater choice**: the BERT-250:1 speaker-window
construal survives (its R=1 de-saturation was the fix for the
listener-side ceiling), but the causal-efficiency result strengthens
the case that causal raters are the natural speaker models — and gives
the paper a clean mechanistic account of *why* bidirectional raters
misbehave on speaker-side questions. Figure:
`analysis/locality/figures/causal_backward.png`.

## Addendum 2 (2026-08-24): combining gpt2m-backward with BERT-forward

Question (Thomas): can we combine gpt2-medium backward surprisal with
BERT forward surprisal? Yes — both halves are banked on the frozen
sample; combination is score-level (raw sum ≈ unnormalized product of
experts; z-sum = scale-free rank blend). Metrics:
`analysis/locality/composite_metrics.csv`; per-instance components:
`composite_components.parquet`; figure:
`figures/composite_frontier.png`.

1. **The two streams are additive, not redundant.** Predicting the
   bidirectional ceiling: R² = 0.095 (backward alone), 0.139 (forward
   R250 alone), **0.259 combined** — slightly super-additive. History
   and future carry non-overlapping information about the pronoun
   (consistent with Finding 3's rank dissociation). Absolute R² is
   depressed by the ceiling's 57% tie mass.
2. **The composite is a construal dial.** α·z(bwd) + (1−α)·z(fwd)
   traces a frontier from pure speaker (α=1: ρ_clean=0.761) to pure
   listener (α=0: ρ_ceiling=0.520 — deep-forward alone is already the
   best cheap proxy for the bidirectional reader). The equal-weight
   point sits at ρ≈0.52/0.51 — balanced, and fully de-saturated
   (tie mass ≈ 0% vs 9.9% for BERT-250:1, 57% for bidirectional).
   The frozen v4 rater plots strictly inside this frontier.
3. **Selection consequence — none, unless deliberately chosen.** The
   composite correlates only ρ≈0.33 with the frozen v4 rater, so
   adopting it is a different manipulation, not a refinement: it would
   mean a full-corpus gpt2m backward pass + a BERT forward-only pass,
   selection v5, recomposing all 95 cells, and re-verification (~1
   day). Banked here as convergent measure + paper material; v4 stays
   frozen.
