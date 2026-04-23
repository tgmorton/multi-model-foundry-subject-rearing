"""
Shared WandB initialization.

Every pod that logs to WandB should call ``init_wandb(config, identity)``
instead of ``wandb.init(...)`` directly. This centralizes how we name,
group, tag, and route runs — so the UI looks consistent whether the run
came from cli.py, the sweep agent, or any future launcher.

Organizational scheme (see also CLAUDE.md):

- **Entity**: ``LoggingConfig.wandb_entity`` or user's default (unset).
- **Project**: routed by ``identity["run_kind"]``:
    - ``production`` → ``LoggingConfig.wandb_project``
      (baseline keeps the existing ``just-drop-the-subject``; new
      production configs should set ``subject-drop-production``).
    - ``hp_sweep`` → ``LoggingConfig.wandb_project_sweeps``
      (default ``subject-drop-sweeps``).
    - Anything else → the production project.
- **Name**: deterministic ``run_id`` so the WandB run name matches the
  registry record key. Reruns of the same cell overwrite.
- **Group**: ``<arch>-<lang>-<condition>`` so WandB's "group by group"
  view collapses all 30 seeds of a cell into one mean±std trajectory.
- **Job type**: ``train`` for training runs, ``sweep`` for HP trials.
- **Tags**: flat ``arch=…``, ``lang=…``, ``condition=…``, ``seed=…``,
  ``run_kind=…`` so cross-cell filters work without opening each run.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _resolve_project(config, identity: Dict[str, Any]) -> str:
    run_kind = identity.get("run_kind", "production")
    if run_kind == "hp_sweep":
        return (
            getattr(config.logging, "wandb_project_sweeps", None)
            or "subject-drop-sweeps"
        )
    return config.logging.wandb_project


def _resolve_group(identity: Dict[str, Any]) -> str:
    return f"{identity['arch']}-{identity['lang']}-{identity['condition']}"


def _resolve_job_type(identity: Dict[str, Any]) -> str:
    kind = identity.get("run_kind", "production")
    if kind == "hp_sweep":
        return "sweep"
    return "train"


def _resolve_tags(identity: Dict[str, Any]) -> list[str]:
    tags = [
        f"arch={identity['arch']}",
        f"lang={identity['lang']}",
        f"condition={identity['condition']}",
        f"seed={identity['seed']}",
        f"run_kind={identity.get('run_kind', 'production')}",
    ]
    return tags


def init_wandb(
    config,
    identity: Dict[str, Any],
    *,
    resume_id: Optional[str] = None,
) -> Optional[str]:
    """
    Initialize WandB for the current run. Returns the resulting
    ``wandb.run.id`` so the caller can write it into the registry
    record. Returns ``None`` if WandB is disabled in the config.

    ``identity`` is the same dict ``cli.py::_registry_identity`` produces
    (run_id / arch / lang / condition / seed / optional run_kind).

    ``resume_id`` is the WandB run-id read from a checkpoint's
    ``metadata.json`` when resuming. When present we resume that same
    WandB run; otherwise WandB generates a fresh id.
    """
    if not config.logging.use_wandb:
        return None

    import wandb  # local import: wandb not available in some test contexts

    project = _resolve_project(config, identity)
    entity = getattr(config.logging, "wandb_entity", None)
    group = _resolve_group(identity)
    job_type = _resolve_job_type(identity)
    tags = _resolve_tags(identity)

    # The deterministic run name matches the registry's run_id so you
    # can click between the WandB run and the registry JSON by identifier.
    name = identity["run_id"]

    # If a sweep agent invoked this (wandb sweep), the WANDB_SWEEP_ID
    # env var is set by the agent and wandb.init picks it up
    # automatically — we don't need to pass it explicitly.

    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=group,
        job_type=job_type,
        tags=tags,
        config=config.model_dump(),
        resume="allow",
        id=resume_id or wandb.util.generate_id(),
    )
    logger.info(
        "wandb.init ok: project=%s group=%s job_type=%s run=%s id=%s",
        project, group, job_type, name, run.id if run else "?",
    )
    return run.id if run else None
