#!/usr/bin/env python3
"""Rater-sweep mode for ``scripts/sweep_agent_lm.py`` (2026-10-16).

HP sweeps for the masked-LM "rater" models (see
memory/project_custom_rater_plan.md, docs/RATER_SWEEPS.md). They differ from
the study-model sweeps in three ways, all handled here so the agent's
existing epoch-horizon path stays untouched:

- **Step budget, not epochs.** Each trial trains ``sweep_train_steps``
  optimizer steps with ``train_steps == scheduler_train_steps``, so the LR
  schedule completes inside the trial.
- **Step-cadence eval + plateau stop.** Held-out MLM perplexity every
  ``eval_every_steps`` on at most ``eval_max_sequences`` test sequences;
  stop once ``patience_evals`` consecutive evals fail to improve the
  reference by ``min_rel_improvement`` (relative), but never before
  ``min_steps_before_stop``. A stopped trial finishes NORMALLY so W&B
  Bayes still learns from it. (Hyperband stays off: it killed winners,
  commit 38621840.)
- **One checkpoint.** No anchors, no resume states, no rolling resume, no
  duplicate ``final/`` copy: the endpoint guard writes a single
  weights-only checkpoint (or nothing with ``save_final: false``).

Switched on by a ``rater_sweep:`` block with ``enabled: true`` in the
baseline YAML, or env ``RATER_SWEEP=1``. Condition and corpus come from env
``CONDITION`` / ``CORPUS_DIR``.

Everything here is pure (no torch / wandb imports) so it is unit-testable;
``sweep_agent_lm.py`` does the I/O.

CLI (used by the precache and sweep K8s jobs):

    python scripts/rater_sweep.py render-config \\
        --base configs/sweeps/baselines/rater_bert_large_en.yaml \\
        --corpus-dir /mnt/data/rater_corpus/sweep_subset_v1/base \\
        --out /tmp/rater_base.yaml

writes the baseline with the corpus swapped in and the ``rater_sweep``
block removed — exactly the (corpus, tokenizer, seq_len, manipulation)
inputs the sweep trial hashes into its cache key.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

RATER_BLOCK_KEY = "rater_sweep"

# The five option-B rater corpora (base + four interventions). Shared with
# scripts/rater_corpus/build_sweep_subset.py.
KNOWN_CORPORA = (
    "base",
    "remove_expletive_sentences_en",
    "impoverish_case_en",
    "lemmatize_verbs",
    "enrich_verbal_morphology",
)

_TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RaterSweepSettings:
    """Per-trial knobs. Defaults are documented in
    configs/sweeps/baselines/rater_bert_large_en.yaml."""

    sweep_train_steps: int = 3000
    eval_every_steps: int = 250
    eval_max_sequences: int = 2000
    patience_evals: int = 4
    min_steps_before_stop: int = 1000
    min_rel_improvement: float = 0.002
    save_final: bool = True
    # D15 (docs/BERT_RECIPE_V2.md). LR-schedule horizon; None = the trial's
    # own budget (schedule completes inside the trial). Set it to the real
    # run's step count to race the production warmup + peak LR, truncated.
    scheduler_train_steps: Optional[int] = None
    # Escape gates: ((step, max_heldout_loss_nats), ...). Every eval at or
    # after `step` must have held-out masked loss ln(ppl) below the max, or
    # the trial stops as "gated out" (still stuck on the MLM plateau).
    gates: tuple = ()

    def __post_init__(self) -> None:
        for name in ("sweep_train_steps", "eval_every_steps",
                     "eval_max_sequences", "patience_evals"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                raise ValueError(f"rater_sweep.{name} must be an int >= 1, got {v!r}")
        if (not isinstance(self.min_steps_before_stop, int)
                or isinstance(self.min_steps_before_stop, bool)
                or self.min_steps_before_stop < 0):
            raise ValueError(
                "rater_sweep.min_steps_before_stop must be an int >= 0, "
                f"got {self.min_steps_before_stop!r}")
        if not 0.0 <= float(self.min_rel_improvement) < 1.0:
            raise ValueError(
                "rater_sweep.min_rel_improvement must be in [0, 1), "
                f"got {self.min_rel_improvement!r}")
        if not isinstance(self.save_final, bool):
            raise ValueError(f"rater_sweep.save_final must be a bool, got {self.save_final!r}")
        if self.scheduler_train_steps is not None and (
                not isinstance(self.scheduler_train_steps, int)
                or isinstance(self.scheduler_train_steps, bool)
                or self.scheduler_train_steps < self.sweep_train_steps):
            raise ValueError(
                "rater_sweep.scheduler_train_steps must be an int >= sweep_train_steps, "
                f"got {self.scheduler_train_steps!r}")
        for g in self.gates:
            if (len(g) != 2 or not isinstance(g[0], int) or isinstance(g[0], bool)
                    or g[0] < 1 or not float(g[1]) > 0):
                raise ValueError(f"rater_sweep.gates entries must be [step >= 1, max_loss > 0], got {g!r}")


def _truthy(value) -> bool:
    return str(value or "").strip().lower() in _TRUTHY


def rater_mode_enabled(cfg: Mapping, env: Mapping) -> bool:
    """True when the baseline's ``rater_sweep.enabled`` or env RATER_SWEEP asks."""
    block = cfg.get(RATER_BLOCK_KEY) or {}
    return _truthy(env.get("RATER_SWEEP")) or block.get("enabled") is True


def load_settings(cfg: Mapping) -> RaterSweepSettings:
    """Parse the ``rater_sweep`` block (missing keys take the defaults).

    Unknown keys are an error, so a typo (``patience_eval``) cannot silently
    fall back to a default.
    """
    block = dict(cfg.get(RATER_BLOCK_KEY) or {})
    block.pop("enabled", None)
    known = {f.name for f in dataclasses.fields(RaterSweepSettings)}
    unknown = sorted(set(block) - known)
    if unknown:
        raise ValueError(f"unknown rater_sweep keys {unknown}; known: {sorted(known)}")
    if "gates" in block:
        block["gates"] = tuple((int(st), float(mx)) for st, mx in (block["gates"] or ()))
    return RaterSweepSettings(**block)


def resolve_condition(env: Mapping, wandb_config: Mapping) -> str:
    """Condition label: env CONDITION, else the sweep's ``condition``
    constant, else "baseline". Both set and different is an error — a pod
    pointed at the wrong sweep would pollute that sweep's posterior."""
    env_cond = str(env.get("CONDITION") or "").strip()
    wc_cond = str(wandb_config.get("condition") or "").strip()
    if env_cond and wc_cond and env_cond != wc_cond:
        raise RuntimeError(
            f"CONDITION={env_cond!r} but the sweep fixes condition={wc_cond!r}; "
            "this pod is pointed at the wrong sweep")
    return env_cond or wc_cond or "baseline"


def check_corpus_matches_condition(corpus_dir: Optional[str], condition: str) -> None:
    """Refuse a CORPUS_DIR whose leaf is a DIFFERENT known corpus.

    The subset builder lays corpora out as ``<out>/<corpus>/``; a custom
    leaf name is allowed, but ``.../lemmatize_verbs`` for condition
    ``base`` is the mis-pairing this guards against.
    """
    if not corpus_dir:
        return
    leaf = Path(corpus_dir.rstrip("/")).name
    if leaf in KNOWN_CORPORA and leaf != condition:
        raise RuntimeError(
            f"CORPUS_DIR={corpus_dir!r} is the {leaf!r} corpus but CONDITION is "
            f"{condition!r}")


def wandb_group(condition: str) -> str:
    return f"rater-sweep-{condition}"


def group_from_env(env: Mapping) -> Optional[str]:
    """W&B group to pass to ``wandb.init`` (groups cannot be changed after
    init), or None outside rater mode / without CONDITION."""
    condition = str(env.get("CONDITION") or "").strip()
    if _truthy(env.get("RATER_SWEEP")) and condition:
        return wandb_group(condition)
    return None


def registry_condition(condition: str) -> str:
    """Registry / W&B-tag condition. Prefixed so rater runs never share a
    condition key with study cells (cf. rater_agent's ``rater_fold_a``)."""
    return f"rater_{condition}"


def endpoint_checkpoint_mode(settings: RaterSweepSettings) -> str:
    """Value for ``Trainer.train(endpoint_checkpoint=...)``."""
    return "weights" if settings.save_final else "none"


def apply_corpus_override(cfg: dict, corpus_dir: Optional[str]) -> None:
    """Point training, source and test corpus at ``corpus_dir`` (in place).

    The subset dir holds ``shard_NNN.train`` and ``heldout.test`` side by
    side; corpus discovery is top-level-only, ``*.train`` for training and
    ``*.test`` for the held-out split.
    """
    if not corpus_dir:
        return
    for key in ("training_corpus", "source_corpus", "test_corpus"):
        cfg["data"][key] = corpus_dir


def apply_rater_overrides(
    cfg: dict,
    settings: RaterSweepSettings,
    *,
    wandb_run_id: str,
    condition: str,
    corpus_dir: Optional[str],
    num_chunks: Callable[[dict], int],
) -> None:
    """Rater-mode counterpart of ``sweep_agent_lm._apply_sweep_overrides``.

    In place. ``num_chunks(cfg)`` returns the chunked-cache row count for
    the (already corpus-overridden) config; it sizes ``epochs`` so the epoch
    loop never ends before ``train_steps``.
    """
    from model_foundry.checkpoint_schedule import compute_opt_steps_per_epoch

    cfg.pop(RATER_BLOCK_KEY, None)
    cfg["experiment_name"] = f"sweep-{wandb_run_id}"
    apply_corpus_override(cfg, corpus_dir)

    t = cfg["training"]
    t["output_dir"] = f"models/rater_sweeps/{condition}/{wandb_run_id}"

    n = settings.sweep_train_steps
    t["train_steps"] = n
    # LR schedule completes inside the trial, unless a longer horizon is set
    # (then the trial is a truncated copy of the real run's schedule).
    t["scheduler_train_steps"] = settings.scheduler_train_steps or n
    t["warmup_steps"] = None           # -> warmup_ratio * scheduler_train_steps

    steps_per_epoch = compute_opt_steps_per_epoch(
        num_chunks=int(num_chunks(cfg)),
        phys_batch=int(cfg["data"]["batch_size"]),
        grad_accum=int(t.get("gradient_accumulation_steps", 1)),
    )
    t["epochs"] = max(1, math.ceil(n / steps_per_epoch))

    # Exactly one checkpoint: the endpoint guard's (weights-only, or none).
    t["checkpoint_schedule"] = []
    t["auto_generate_checkpoints"] = False
    t["resume_from_checkpoint"] = False
    t["resume_state_steps"] = None
    t["rolling_resume_every_steps"] = None


# ---------- early stop ----------

@dataclass(frozen=True)
class EarlyStopState:
    """Plateau-stop bookkeeping across evals of one trial.

    ``min_ppl`` is the true running minimum (what W&B Bayes ranks on).
    ``ref_ppl`` only moves on a SIGNIFICANT improvement, so a slow creep of
    sub-threshold gains still counts toward patience (Keras ``min_delta``
    semantics, relative instead of absolute).
    """

    min_ppl: float = math.inf
    best_step: int = -1
    ref_ppl: float = math.inf
    bad_evals: int = 0
    n_evals: int = 0
    should_stop: bool = False


def early_stop_update(
    state: EarlyStopState, step: int, ppl: float, settings: RaterSweepSettings,
) -> EarlyStopState:
    """Fold one eval into the state; pure.

    An eval improves when it beats the reference by more than
    ``min_rel_improvement`` (relative); gains are cumulative against the
    reference, so a steady slow improvement still resets patience every few
    evals while a creep that stays under the threshold for
    ``patience_evals`` evals stops. A non-finite ``ppl`` counts as a
    non-improving eval. Bad evals accumulate from the start, but
    ``should_stop`` is only raised at ``step >= min_steps_before_stop``.
    """
    finite = ppl is not None and math.isfinite(ppl)
    min_ppl, best_step = state.min_ppl, state.best_step
    if finite and ppl < min_ppl:
        min_ppl, best_step = ppl, step
    if finite and ppl < state.ref_ppl * (1.0 - settings.min_rel_improvement):
        ref_ppl, bad = ppl, 0
    else:
        ref_ppl, bad = state.ref_ppl, state.bad_evals + 1
    should_stop = (step >= settings.min_steps_before_stop
                   and bad >= settings.patience_evals)
    return EarlyStopState(
        min_ppl=min_ppl, best_step=best_step, ref_ppl=ref_ppl,
        bad_evals=bad, n_evals=state.n_evals + 1, should_stop=should_stop,
    )


def gate_failure(step: int, ppl: float, settings: RaterSweepSettings):
    """The first escape gate this eval fails, as (gate_step, max_loss), or None.

    A gate applies to every eval at or after its step; held-out masked loss
    is ln(ppl) in nats, and a non-finite ppl fails any gate that applies."""
    finite = ppl is not None and math.isfinite(ppl) and ppl > 0
    loss = math.log(ppl) if finite else math.inf
    for g_step, max_loss in settings.gates:
        if step >= g_step and loss >= max_loss:
            return (g_step, max_loss)
    return None


def is_eval_step(step: int, settings: RaterSweepSettings) -> bool:
    return step > 0 and step % settings.eval_every_steps == 0


# ---------- CLI ----------

def _cmd_render_config(args: argparse.Namespace) -> int:
    import yaml

    cfg = yaml.safe_load(Path(args.base).read_text())
    if not rater_mode_enabled(cfg, {}):
        print(f"FATAL: {args.base} has no `rater_sweep: {{enabled: true}}` block",
              file=sys.stderr)
        return 2
    load_settings(cfg)  # validate before anything consumes the file
    cfg.pop(RATER_BLOCK_KEY, None)
    apply_corpus_override(cfg, args.corpus_dir)
    Path(args.out).write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"rendered {args.base} (corpus={args.corpus_dir}) -> {args.out}")
    return 0


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rc = sub.add_parser("render-config",
                        help="baseline + corpus override, rater block stripped")
    rc.add_argument("--base", default="configs/sweeps/baselines/rater_bert_large_en.yaml")
    rc.add_argument("--corpus-dir", required=True)
    rc.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "render-config":
        return _cmd_render_config(args)
    return 2


if __name__ == "__main__":
    # Allow `python scripts/rater_sweep.py` from the repo root.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.exit(main())
