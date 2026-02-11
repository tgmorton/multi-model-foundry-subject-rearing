"""Europarl pronoun recovery sweep runner.

Maps a JOB_COMPLETION_INDEX (0-26) to a specific (lr, scale, alpha)
combination and runs training + evaluation.

Grid: 3 learning rates × 3 data scales × 3 alphas = 27 runs.
"""

import json
import logging
import os
import sys
from itertools import product
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Sweep grid ──────────────────────────────────────────────────────────

LEARNING_RATES = [5e-6, 1e-5, 2e-5]
DATA_SCALES = [100_000, 300_000, 500_000]
ALPHAS = [0.0, 0.5, 1.0]

GRID = list(product(LEARNING_RATES, DATA_SCALES, ALPHAS))
assert len(GRID) == 27

# ── Paths ───────────────────────────────────────────────────────────────

DATA_DIR = Path(os.environ.get(
    "SWEEP_DATA_DIR",
    "/mnt/data/pronoun_recovery/europarl_aligned",
))
OUTPUT_BASE = Path(os.environ.get(
    "SWEEP_OUTPUT_DIR",
    "/mnt/data/pronoun_recovery/models/europarl_sweep",
))
TEST_DATA = DATA_DIR / "it_test" / "packed_checkpoint.jsonl"


def get_data_path(scale: int) -> Path:
    return DATA_DIR / "it_train" / f"packed_{scale}.jsonl"


def run_training(lr: float, scale: int, alpha: float, run_idx: int) -> None:
    """Run a single training + evaluation."""
    from analysis.pronoun_recovery.config import TrainingConfig
    from analysis.pronoun_recovery.model.trainer import PronounRecoveryTrainer

    run_name = f"europarl_lr{lr:.0e}_scale{scale//1000}k_alpha{alpha}"
    output_dir = OUTPUT_BASE / run_name
    data_path = get_data_path(scale)

    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        sys.exit(1)

    logger.info(
        "=== Run %d/27: lr=%s, scale=%dk, alpha=%s ===",
        run_idx + 1, lr, scale // 1000, alpha,
    )
    logger.info("Data: %s", data_path)
    logger.info("Output: %s", output_dir)

    config = TrainingConfig(
        training_source="llm_annotated",
        annotation_data_path=str(data_path),
        output_path=str(output_dir),
        language="it",
        model_name="microsoft/mdeberta-v3-base",
        num_labels=7,
        phase_b_lr=lr,
        phase_b_batch_size=16,
        phase_b_epochs=20,
        phase_b_patience=3,
        weight_alpha=alpha,
        focal_gamma=0.0,
        fp16=True,
        warmup_ratio=0.1,
        max_seq_length=256,
        seed=42,
        wandb_project="pronoun-recovery",
        wandb_run_name=run_name,
        min_detection_f1=0.0,   # Don't gate during sweep
        min_label_accuracy=0.0,
    )

    trainer = PronounRecoveryTrainer(config)
    final_model_path = trainer.train()

    # ── Evaluate on held-out test set ──
    logger.info("Evaluating on held-out test set: %s", TEST_DATA)

    from analysis.pronoun_recovery.model.dataset import build_dataset
    from analysis.pronoun_recovery.model.evaluate import ModelEvaluator

    test_dataset = build_dataset(
        data_path=TEST_DATA,
        tokenizer=trainer.tokenizer,
        max_length=config.max_seq_length,
        max_samples=0,
        data_type="checkpoint",
    )

    evaluator = ModelEvaluator(final_model_path, config)
    results = evaluator.evaluate(test_dataset)

    # Save results
    results_path = output_dir / "test_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(
            {
                "run_name": run_name,
                "lr": lr,
                "scale": scale,
                "alpha": alpha,
                "test_results": results,
            },
            f,
            indent=2,
        )
    logger.info("Test results saved to %s", results_path)

    # Log test metrics to wandb if available
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({
                "test/detection_f1": results.get("detection_f1", 0),
                "test/macro_f1": results.get("macro_f1", 0),
                "test/weighted_f1": results.get("weighted_f1", 0),
                "test/feature_accuracy": results.get("feature_accuracy", 0),
            })
    except ImportError:
        pass

    logger.info(
        "Run %d complete: test detection_f1=%.4f, macro_f1=%.4f",
        run_idx + 1,
        results.get("detection_f1", 0),
        results.get("macro_f1", 0),
    )


def main() -> None:
    run_idx = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))

    if run_idx >= len(GRID):
        logger.error(
            "JOB_COMPLETION_INDEX=%d exceeds grid size %d", run_idx, len(GRID)
        )
        sys.exit(1)

    lr, scale, alpha = GRID[run_idx]
    logger.info(
        "Grid index %d → lr=%s, scale=%d, alpha=%s",
        run_idx, lr, scale, alpha,
    )

    run_training(lr, scale, alpha, run_idx)


if __name__ == "__main__":
    main()
