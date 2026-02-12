"""
CLI entry point for the pronoun recovery pipeline.

Usage:
    python -m analysis.pronoun_recovery.run synthetic --config configs/analysis/pronoun_recovery/pronoun_recovery_en_synthetic.yaml
    python -m analysis.pronoun_recovery.run sample --config configs/analysis/pronoun_recovery/pronoun_recovery_en_annotate.yaml
    python -m analysis.pronoun_recovery.run annotate --config configs/analysis/pronoun_recovery/pronoun_recovery_en_annotate.yaml
    python -m analysis.pronoun_recovery.run validate --config configs/analysis/pronoun_recovery/pronoun_recovery_en_annotate.yaml
    python -m analysis.pronoun_recovery.run train --config configs/analysis/pronoun_recovery/pronoun_recovery_en_train.yaml
    python -m analysis.pronoun_recovery.run insert --config configs/analysis/pronoun_recovery/pronoun_recovery_en_insert.yaml
    python -m analysis.pronoun_recovery.run train-seq2seq --config configs/analysis/pronoun_recovery/pronoun_recovery_en_train_seq2seq.yaml
    python -m analysis.pronoun_recovery.run insert-seq2seq --config configs/analysis/pronoun_recovery/pronoun_recovery_en_insert_seq2seq.yaml
"""

import logging
from pathlib import Path

from dotenv import load_dotenv
import typer

load_dotenv()

from .config import (
    EuroparlAlignmentConfig,
    InsertionConfig,
    LLMAnnotationConfig,
    ModelTrainingConfig,
    Seq2SeqInsertionConfig,
    Seq2SeqTrainingConfig,
    SyntheticDataConfig,
    ValidationConfig,
    load_config,
)

app = typer.Typer(
    name="pronoun-recovery",
    help="Pronoun Recovery Model — Phase 2 pipeline.",
)

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@app.command()
def synthetic(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Step 1: Generate synthetic training pairs from corpus."""
    _setup_logging(verbose)
    cfg = load_config(config, SyntheticDataConfig)
    logger.info(f"Generating synthetic pairs from {cfg.input_path}")

    from .synthetic_data.generator import SyntheticPairGenerator

    generator = SyntheticPairGenerator(cfg)
    generator.generate_from_corpus()
    logger.info("Synthetic data generation complete.")


@app.command()
def sample(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Step 2a: Sample candidate sequences for annotation."""
    _setup_logging(verbose)
    cfg = load_config(config, LLMAnnotationConfig)
    logger.info(f"Sampling candidates from {cfg.corpus_path}")

    from .annotation.sampler import AnnotationSampler

    sampler = AnnotationSampler(cfg)
    candidates = sampler.sample_candidates()
    output_path = Path(cfg.output_path) / "candidates.jsonl"
    sampler.save_candidates(candidates, output_path)
    logger.info(f"Saved {len(candidates)} candidates to {output_path}")


@app.command("seed-export")
def seed_export(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config"),
    n: int = typer.Option(15, "--n", "-n", help="Number of seed candidates"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Step 2b: Export seed candidates for manual annotation."""
    _setup_logging(verbose)
    cfg = load_config(config, LLMAnnotationConfig)

    from .annotation.seed_workflow import export_seed_yaml, select_seed_candidates

    candidates_path = Path(cfg.output_path) / "candidates.jsonl"
    if not candidates_path.exists():
        typer.echo(f"Error: {candidates_path} not found. Run 'sample' first.")
        raise typer.Exit(1)

    selected = select_seed_candidates(candidates_path, n=n)
    yaml_path = Path(cfg.output_path) / "manual" / "seed_annotations.yaml"
    export_seed_yaml(selected, yaml_path)
    typer.echo(f"Exported {len(selected)} candidates to {yaml_path}")
    typer.echo("Edit the 'annotated' field in each entry, then run 'seed-import'.")


@app.command("seed-import")
def seed_import(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Step 2c: Import and validate completed seed annotations."""
    _setup_logging(verbose)
    cfg = load_config(config, LLMAnnotationConfig)

    from .annotation.seed_workflow import import_seed_yaml, save_seed_jsonl

    yaml_path = Path(cfg.output_path) / "manual" / "seed_annotations.yaml"
    if not yaml_path.exists():
        typer.echo(f"Error: {yaml_path} not found. Run 'seed-export' first.")
        raise typer.Exit(1)

    try:
        annotations = import_seed_yaml(yaml_path, language=cfg.language)
    except ValueError as e:
        typer.echo(f"Validation failed:\n{e}")
        raise typer.Exit(1)

    seed_path = Path(cfg.output_path) / "manual" / "seed_set.jsonl"
    save_seed_jsonl(annotations, seed_path)
    typer.echo(f"Saved {len(annotations)} validated seed annotations to {seed_path}")


@app.command()
def annotate(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Step 3: Run LLM-scaled annotation via DeepSeek."""
    _setup_logging(verbose)
    cfg = load_config(config, LLMAnnotationConfig)
    logger.info("Starting LLM annotation pipeline")

    import json

    from .annotation.batch_runner import AnnotationBatchRunner

    # Load candidates
    candidates_path = Path(cfg.output_path) / "candidates.jsonl"
    if not candidates_path.exists():
        typer.echo(f"Error: candidates file not found at {candidates_path}. Run 'sample' first.")
        raise typer.Exit(1)

    candidates = []
    with open(candidates_path) as f:
        for line in f:
            candidates.append(json.loads(line))

    # Load seed examples
    seed_examples = []
    seed_path = cfg.seed_set_path or (Path(cfg.output_path) / "manual" / "seed_set.jsonl")
    if seed_path.exists():
        with open(seed_path) as f:
            for line in f:
                seed_examples.append(json.loads(line))
        logger.info(f"Loaded {len(seed_examples)} seed examples from {seed_path}")
    else:
        typer.echo(f"Warning: no seed set found at {seed_path}. Running without few-shot examples.")

    runner = AnnotationBatchRunner(cfg)
    results = runner.run(candidates, seed_examples)
    logger.info(f"Annotation complete: {len(results)} results.")


@app.command()
def validate(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Step 4: Validate LLM annotations against gold standard."""
    _setup_logging(verbose)
    cfg = load_config(config, ValidationConfig)
    logger.info("Running annotation validation")

    import json

    from .validation.metrics import compute_annotation_metrics
    from .validation.reviewer import HumanReviewer

    # Load annotations
    annotations = []
    with open(cfg.annotations_path) as f:
        for line in f:
            annotations.append(json.loads(line))

    # Load gold
    gold = []
    with open(cfg.gold_path) as f:
        for line in f:
            gold.append(json.loads(line))

    metrics = compute_annotation_metrics(annotations, gold)
    logger.info(
        f"Metrics: P={metrics.precision:.3f} R={metrics.recall:.3f} "
        f"F1={metrics.f1:.3f} FA={metrics.feature_accuracy:.3f}"
    )

    reviewer = HumanReviewer(cfg)
    gate_results = reviewer.check_gates(metrics)
    if gate_results["all_pass"]:
        logger.info("All quality gates passed.")
    else:
        logger.warning(f"Quality gates failed: {gate_results}")

    # Generate review sheet
    review_items = reviewer.sample_for_review(annotations)
    review_path = Path(cfg.output_path) / "review_sheet.jsonl"
    reviewer.save_review_sheet(review_items, review_path)
    logger.info(f"Review sheet saved to {review_path}")


@app.command()
def train(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Step 5: Train DeBERTa sequence labeler."""
    _setup_logging(verbose)
    cfg = load_config(config, ModelTrainingConfig)
    logger.info("Starting model training (source=%s)", cfg.training_source)

    from .model.trainer import PronounRecoveryTrainer

    trainer = PronounRecoveryTrainer(cfg)
    best_checkpoint = trainer.train()
    logger.info(f"Training complete. Best checkpoint: {best_checkpoint}")

    # Evaluate
    if cfg.annotation_data_path:
        from .model.evaluate import ModelEvaluator
        from .model.sequence_labeler import PronounRecoveryModel

        model = PronounRecoveryModel(best_checkpoint)
        evaluator = ModelEvaluator(model)

        import json

        eval_data = []
        eval_path = Path(cfg.annotation_data_path) / "eval.jsonl"
        if eval_path.exists():
            with open(eval_path) as f:
                for line in f:
                    eval_data.append(json.loads(line))
            results = evaluator.evaluate(eval_data)
            gate_results = evaluator.check_gates(results, cfg)
            if gate_results["all_pass"]:
                logger.info("All model quality gates passed.")
            else:
                logger.warning(f"Model quality gates: {gate_results}")
            evaluator.save_report(results, Path(cfg.output_path) / "eval_report.json")


@app.command()
def insert(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Stage 2: Run deterministic pronoun insertion on full corpus."""
    _setup_logging(verbose)
    cfg = load_config(config, InsertionConfig)
    logger.info(f"Starting corpus insertion from {cfg.input_path}")

    from .insertion.corpus_processor import CorpusInsertionPipeline

    pipeline = CorpusInsertionPipeline(cfg)
    metadata = pipeline.run()
    logger.info(f"Insertion complete. Output: {cfg.output_path}")
    logger.info(f"Processed {metadata.get('total_lines', '?')} lines.")


@app.command("europarl-align")
def europarl_align(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Generate pronoun labels from EN-IT Europarl parallel alignment."""
    _setup_logging(verbose)
    cfg = load_config(config, EuroparlAlignmentConfig)
    logger.info(
        "Starting Europarl alignment pipeline (%d-%d)",
        cfg.start_line, cfg.end_line,
    )

    from .parallel_data.generator import EuroparlAlignmentGenerator

    generator = EuroparlAlignmentGenerator(cfg)
    generator.run()
    logger.info("Europarl alignment complete.")


@app.command("train-seq2seq")
def train_seq2seq(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Train mT5 seq2seq pronoun recovery model (two-phase)."""
    _setup_logging(verbose)
    cfg = load_config(config, Seq2SeqTrainingConfig)
    logger.info("Starting seq2seq model training")

    from .model.seq2seq_trainer import Seq2SeqPronounTrainer

    trainer = Seq2SeqPronounTrainer(cfg)
    best_checkpoint = trainer.train()
    logger.info(f"Seq2seq training complete. Best checkpoint: {best_checkpoint}")


@app.command("insert-seq2seq")
def insert_seq2seq(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run seq2seq pronoun insertion on full corpus."""
    _setup_logging(verbose)
    cfg = load_config(config, Seq2SeqInsertionConfig)
    logger.info(f"Starting seq2seq corpus insertion from {cfg.input_path}")

    from .insertion.seq2seq_processor import Seq2SeqInsertionPipeline

    pipeline = Seq2SeqInsertionPipeline(cfg)
    metadata = pipeline.run()
    logger.info(f"Seq2seq insertion complete. Output: {cfg.output_path}")
    logger.info(f"Processed {metadata.get('total_lines', '?')} lines.")


if __name__ == "__main__":
    app()
