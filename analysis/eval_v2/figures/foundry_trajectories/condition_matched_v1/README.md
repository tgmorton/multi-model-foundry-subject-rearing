# Condition-matched Foundry trajectory figures

All panels in this folder use `null_subj_v2_condition_matched_v1` and the regular binary overt-preference measure derived from length-normalized mean log likelihood. Lines aggregate equally over training cells unless a title says seed; ribbons are ±1 SE. The x-axis is tokens seen: each training figure begins at its earliest evaluated checkpoint, and each architecture is hard-truncated at the smallest terminal token count among its five interventions (`final_token_cutoffs.csv`). Only observed checkpoints at or below that boundary are retained; no endpoint interpolation is performed. Full-horizon panels mark the boundary with a vertical dashed line. No SLOR results and no old generic-evaluation values are mixed into these panels.

Checkpoint −1 is kept separate and only shown for architectures/seeds with audited `null_subj_v2_condition_matched_init_v1` results. See `coverage_manifest.json` for panels that cannot yet be reproduced without missing matched evaluations. Remove-expletive stimuli are intentionally unchanged under the approved training-deprivation estimand; morphology enrichment uses the approved literal transformed stimuli.

## End-state forests

See `endstate_forests/` for architecture-specific, token-matched intervention comparisons across all 24 evaluation conditions.
