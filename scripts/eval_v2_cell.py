#!/usr/bin/env python3
"""Production per-run null-subject eval — one (arch, lang, condition, hp, seed) run.

Resolves everything from a registry run_id (e.g.
``gpt2_small-en-baseline-h0-s42``): checkpoint root, tokenizer, model
factory, scorer. Runs the v2 pipeline (stimuli cache → load-once/swap
runner → flat parquet, with D11 idempotency markers) over EVERY
checkpoint of the run, then writes a sidecar checkpoint-metadata parquet
(step → tokens_seen) and updates the run registry.

Per-architecture handling:
  gpt2_*      AutoModelForCausalLM from the checkpoint's own config.json;
              causal batched surprisal.
  bert_large  AutoModelForMaskedLM + PLL scoring (Salazar et al. 2020) —
              left-to-right surprisal is not defined for a bidirectional
              MLM.
  lstm,       Foundry model classes rebuilt from the same baseline config
  mamba_370m  the production agent trained from
              (configs/sweeps/baselines/{arch}_{lang}.yaml). HP-winner
              overlays only touch optimizer HPs + dropout, neither of
              which matters under model.eval(), so the baseline yaml
              fully determines the architecture. Vocab is synced to the
              tokenizer exactly like Trainer._sync_vocab_size_with_tokenizer.

Faithfulness guards baked in:
  - tokenizer probe: for SP-unigram archs, the raw SentencePiece encoding
    is cross-checked against the training-side load_tokenizer() encoding
    on real stimulus text; any id mismatch aborts the run.
  - UNK-rate gate: if >1% of stimulus tokens are UNK the run warns,
    >10% aborts (catches casing/preprocessing mismatches).
  - state-dict swap raises on any missing/unexpected/mis-shaped key
    (per_model_runner), so a wrong factory config cannot silently
    produce random-weight scores.

Usage (inside an eval pod with the PVC at /mnt/data):
    python scripts/eval_v2_cell.py --run_id gpt2_small-en-baseline-h0-s42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("eval_v2_cell")

DEFAULT_BENCHMARK = "null_subj_v2"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_matched_stimuli(root: Path, expected_sha256: str | None) -> tuple[dict, str]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing matched-stimulus manifest: {manifest_path}")
    digest = _sha256_file(manifest_path)
    if expected_sha256 and digest != expected_sha256:
        raise SystemExit(
            f"matched-stimulus manifest SHA mismatch: {digest} != {expected_sha256}")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("format_version") != "condition-matched-stimuli.v1" or
            manifest.get("vetted") is not True):
        raise SystemExit("matched-stimulus manifest is not a vetted v1 artifact")
    for condition, entry in manifest.get("conditions", {}).items():
        for name, expected in entry.get("output_sha256", {}).items():
            path = root / condition / "en" / name
            if not path.is_file() or _sha256_file(path) != expected:
                raise SystemExit(f"matched-stimulus content mismatch: {path}")
    return manifest, digest

# Tokenizer directory per (arch, lang) — mirrors configs/sweeps/baselines/*.yaml.
def _tokenizer_dirname(arch: str, lang: str) -> str:
    if arch == "bert_large":
        return f"{lang}_bert_wordpiece"
    return f"{lang}_shared_unigram"


def parse_run_id(run_id: str):
    """``{arch}-{lang}-{condition}-h{hp}-s{seed}`` → parts.

    arch/condition use underscores internally, so '-' splits cleanly.
    """
    parts = run_id.split("-")
    if len(parts) != 5 or not parts[3].startswith("h") or not parts[4].startswith("s"):
        raise ValueError(f"Unparseable run_id: {run_id!r}")
    arch, lang, condition = parts[0], parts[1], parts[2]
    return arch, lang, condition, int(parts[3][1:]), int(parts[4][1:])


class HFEvalTokenizer:
    """Adapter giving an HF tokenizer the SP-processor API the stimulus
    cache expects (`encode` without specials, `bos_id`, `eos_id`,
    `get_piece_size`).

    For BERT, bos maps to [CLS] (prepended to every sequence by the
    cache); [SEP] is appended by the PLL scorer at scoring time.
    """

    def __init__(self, hf_tokenizer):
        self.hf = hf_tokenizer

    def encode(self, text: str):
        return self.hf.encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self.hf.decode(list(ids), skip_special_tokens=False)

    def bos_id(self):
        v = getattr(self.hf, "cls_token_id", None)
        if v is None:
            v = getattr(self.hf, "bos_token_id", None)
        return v if v is not None else -1

    def eos_id(self):
        v = getattr(self.hf, "sep_token_id", None)
        if v is None:
            v = getattr(self.hf, "eos_token_id", None)
        return v if v is not None else -1

    def get_piece_size(self):
        return len(self.hf)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--data_root", default="/mnt/data")
    ap.add_argument("--stimuli_dir",
                    default=str(REPO_ROOT / "evaluation/stimuli/null-subj-v2/staging"),
                    help="Root containing per-language stimulus CSV dirs.")
    ap.add_argument(
        "--matched_stimuli_root",
        help=("Condition-matched root laid out as <root>/<condition>/<lang>/*.csv. "
              "When set, this takes precedence over --stimuli_dir."),
    )
    ap.add_argument("--expected_stimuli_manifest_sha256")
    ap.add_argument("--output_root", default="/mnt/data/eval_v2/null_subj_v2")
    ap.add_argument(
        "--benchmark",
        default=DEFAULT_BENCHMARK,
        help="Registry/S3 benchmark key. Use a new value for a new stimulus regime.",
    )
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--scoring_version", default="null-subj-v2-r1")
    ap.add_argument("--scratch_dir", help="Pod-local staging dir for parquet writes.")
    ap.add_argument("--prefetch_dir", default="/tmp/eval_prefetch",
                    help="Pod-local staging dir for checkpoint reads — each "
                         "checkpoint crosses the shared FS once, in the "
                         "background, while the GPU scores the previous one. "
                         "Empty string disables.")
    ap.add_argument("--prefetch_depth", type=int, default=3)
    ap.add_argument("--slor", action="store_true",
                    help="Load the per-condition unigram baseline (from S3) "
                         "so the runner emits SLOR alongside MeanLP. "
                         "Skipped for bert_large (no wordpiece unigram yet).")
    ap.add_argument("--results-to-pvc", action="store_true",
                    help="Also copy result parquets to output_root on the "
                         "PVC (legacy behavior). Default is S3-only: the "
                         "slow CephFS parquet write — the GPU-idle tail — "
                         "is skipped; S3 is the canonical results store "
                         "and resume integrity reads S3 when needed.")
    ap.add_argument("--no-registry", action="store_true",
                    help="Skip S3 registry writes (local/smoke runs).")
    ap.add_argument("--rerun", action="store_true",
                    help="Drop existing D11 markers for this cell first.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    arch, lang, condition, hp_rank, seed = parse_run_id(args.run_id)
    log.info("run_id=%s → arch=%s lang=%s condition=%s hp=%d seed=%d",
             args.run_id, arch, lang, condition, hp_rank, seed)

    import torch

    from evaluation.cache import CachedRunner
    from evaluation.runners.per_model_runner import (
        CellSpec, find_checkpoints_sorted,
    )
    from evaluation.stimuli_cache import build_or_load_cache

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    log.info("device=%s", device)

    data_root = Path(args.data_root)
    ckpt_root = data_root / "models" / "production" / args.run_id
    tokenizer_dir = data_root / "tokenizers" / _tokenizer_dirname(arch, lang)
    output_root = Path(args.output_root)
    matched_manifest = None
    matched_manifest_sha = None
    if args.matched_stimuli_root:
        matched_root = Path(args.matched_stimuli_root)
        if args.benchmark == DEFAULT_BENCHMARK:
            raise SystemExit("matched stimuli require a distinct non-legacy benchmark")
        matched_manifest, matched_manifest_sha = _validate_matched_stimuli(
            matched_root, args.expected_stimuli_manifest_sha256)
        stimuli_dir = matched_root / condition / lang
    else:
        if args.benchmark != DEFAULT_BENCHMARK:
            raise SystemExit("non-legacy benchmark requires --matched_stimuli_root")
        stimuli_dir = Path(args.stimuli_dir, lang)
    stimuli_paths = sorted(stimuli_dir.glob("*.csv"))

    if not stimuli_paths:
        log.error("No stimulus CSVs under %s", stimuli_dir)
        sys.exit(2)
    for p in (ckpt_root, tokenizer_dir):
        if not p.exists():
            log.error("Missing path: %s", p)
            sys.exit(2)

    ckpts = find_checkpoints_sorted(ckpt_root)
    if not ckpts:
        log.error("No checkpoints under %s", ckpt_root)
        sys.exit(3)
    log.info("%d checkpoint(s), steps %d..%d",
             len(ckpts), ckpts[0][0], ckpts[-1][0])
    if len(ckpts) < 60:
        log.warning("Only %d checkpoints — production runs should have ~80. "
                    "Proceeding (registry records the actual count).", len(ckpts))

    # ── Tokenizer ────────────────────────────────────────────────────────
    # Training-side loader: same code path the trainer used, so len()
    # reproduces the post-special-tokens vocab the embeddings were sized to.
    from model_foundry.training.tokenization import load_tokenizer
    train_tok = load_tokenizer(str(tokenizer_dir))
    actual_vocab = len(train_tok)
    log.info("training-side tokenizer: len=%d", actual_vocab)

    if arch == "bert_large":
        cache_tok = HFEvalTokenizer(train_tok)
        hash_file = next(
            (p for p in (tokenizer_dir / "tokenizer.json",
                         tokenizer_dir / "vocab.txt",
                         tokenizer_dir / "tokenizer.model") if p.exists()),
            None,
        )
        pad_id = train_tok.pad_token_id
        mask_id = train_tok.mask_token_id
        sep_id = train_tok.sep_token_id
        if mask_id is None or pad_id is None:
            log.error("BERT tokenizer lacks mask/pad ids — cannot PLL-score.")
            sys.exit(2)
    else:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        hash_file = tokenizer_dir / "tokenizer.model"
        sp.load(str(hash_file))
        cache_tok = sp
        pad_id = sp.pad_id() if sp.pad_id() >= 0 else max(sp.unk_id(), 0)
        # Probe: raw-SP ids must match the training-side loader's ids on
        # real stimulus text (specials stripped). An id permutation here
        # would make every score garbage while staying finite.
        import csv as _csv
        with open(stimuli_paths[0], newline="", encoding="utf-8") as fh:
            rows = list(_csv.DictReader(fh))[:5]
        for r in rows:
            probe = (r.get("context") or "") + " " + (r.get("target") or "")
            probe = probe.strip()
            if not probe:
                continue
            sp_ids = list(sp.encode(probe))
            try:
                hf_ids = train_tok.encode(probe, add_special_tokens=False)
            except TypeError:  # SP wrapper without HF kwargs
                hf_ids = list(train_tok.encode(probe))
            if sp_ids != list(hf_ids):
                log.error("Tokenizer probe mismatch!\n  text: %r\n  sp:  %s\n  hf:  %s",
                          probe, sp_ids[:20], list(hf_ids)[:20])
                sys.exit(4)
        log.info("tokenizer probe: raw-SP == training-side loader ✓")

    if hash_file is None:
        log.error("No hashable tokenizer file in %s", tokenizer_dir)
        sys.exit(2)

    # ── Stimuli cache (D6) ──────────────────────────────────────────────
    cache_dir = output_root / "stimuli_cache"
    stimuli = build_or_load_cache(stimuli_paths, cache_tok, hash_file, cache_dir)
    log.info("stimuli: n=%d languages=%s categories=%s",
             len(stimuli), stimuli.meta.get("languages"),
             stimuli.meta.get("categories"))

    # UNK-rate gate.
    unk_id = (train_tok.unk_token_id if arch == "bert_large"
              else (cache_tok.unk_id() if cache_tok.unk_id() >= 0 else None))
    if unk_id is not None:
        total = sum(len(e.full_token_ids) for e in stimuli.rows)
        n_unk = sum(sum(1 for t in e.full_token_ids if t == unk_id)
                    for e in stimuli.rows)
        rate = n_unk / max(total, 1)
        log.info("UNK rate over stimuli: %.4f%% (%d/%d)", 100 * rate, n_unk, total)
        if rate > 0.10:
            log.error("UNK rate >10%% — tokenizer/stimuli mismatch. Aborting.")
            sys.exit(4)
        if rate > 0.01:
            log.warning("UNK rate >1%% — check stimulus preprocessing.")

    # ── Model factory + scorer ──────────────────────────────────────────
    scorer = None
    # Training configs carry attn_implementation=flash_attention_2, but FA2
    # rejects fp32 inputs and eval runs in full fp32 (no autocast). Eager
    # attention is the exact same math (transformers 4.41 in the image has
    # no GPT-2 SDPA support); throughput is irrelevant at ~1k stimuli.
    if arch.startswith("gpt2"):
        from transformers import AutoConfig, AutoModelForCausalLM
        hf_config = AutoConfig.from_pretrained(str(ckpts[0][1]))

        def model_factory():
            return AutoModelForCausalLM.from_config(
                hf_config, attn_implementation="eager")

    elif arch == "bert_large":
        from transformers import AutoConfig, AutoModelForMaskedLM
        from evaluation.core.pll_surprisal import make_pll_scorer
        hf_config = AutoConfig.from_pretrained(str(ckpts[0][1]))
        scorer = make_pll_scorer(mask_id=mask_id, sep_id=sep_id)

        def model_factory():
            return AutoModelForMaskedLM.from_config(
                hf_config, attn_implementation="eager")

    elif arch in ("lstm", "mamba_370m"):
        from model_foundry.cli import load_config
        from model_foundry.model import create_model
        base_cfg = REPO_ROOT / "configs" / "sweeps" / "baselines" / f"{arch}_{lang}.yaml"
        cfg = load_config(str(base_cfg))
        if cfg.tokenizer.vocab_size != actual_vocab:
            log.info("vocab sync: %d → %d", cfg.tokenizer.vocab_size, actual_vocab)
            cfg.tokenizer.vocab_size = actual_vocab

        def model_factory():
            return create_model(cfg)

    else:
        log.error("No eval factory for arch %r", arch)
        sys.exit(2)

    # ── Unigram baseline (SLOR) ─────────────────────────────────────────
    # Per-condition table (FIT-CLAMS): the run is normalized against the
    # unigram statistics of ITS OWN training corpus, so SLOR factors out
    # each model's corpus-specific token-frequency expectations — the
    # confound the hotspot analysis exposed in MeanLP. Resolved by
    # (lang, condition, tokenizer-family); bert (wordpiece) has no SP
    # table yet, so it stays MeanLP-only.
    unigram = None
    if args.slor and arch != "bert_large":
        from evaluation.unigram import UnigramTable
        uni_name = f"{lang}_{condition}__shared_unigram.pkl"
        uni_local = Path("/tmp") / uni_name
        if not uni_local.exists():
            import boto3
            bucket = os.environ.get("REGISTRY_BUCKET")
            s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"))
            s3.download_file(bucket, f"unigrams/{uni_name}", str(uni_local))
        unigram = UnigramTable.load(uni_local)
        # The unigram's tokenizer must match the stimuli's tokenizer.
        if unigram.tokenizer_id != stimuli.tokenizer_id:
            log.error("unigram tokenizer_id %s != stimuli tokenizer_id %s",
                      unigram.tokenizer_id, stimuli.tokenizer_id)
            sys.exit(4)
        log.info("SLOR on: unigram %s (V=%d N=%d)", uni_name,
                 unigram.vocab_size, unigram.total_tokens)

    cell = CellSpec(
        cell_id=args.run_id,
        architecture=arch,
        intervention=condition,
        rep=seed,
        checkpoint_root=ckpt_root,
        model_factory=model_factory,
        stimuli=stimuli,
        unigram=unigram,
        device=device,
        batch_size=args.batch_size,
        scorer=scorer,
        pad_id=pad_id,
    )

    # ── Registry: benchmark start ───────────────────────────────────────
    reg_kwargs = dict(run_id=args.run_id, arch=arch, lang=lang,
                      condition=condition, benchmark=args.benchmark)
    if not args.no_registry:
        from model_foundry import registry
        registry.register_eval_start(**reg_kwargs)

    if args.rerun:
        marker_dir = output_root / ".cache"
        if marker_dir.exists():
            for m in marker_dir.glob(
                    f"{args.run_id}__*__scoring={args.scoring_version}.done"):
                m.unlink()
                log.info("dropped marker %s", m.name)

    # ── Run ─────────────────────────────────────────────────────────────
    def _remote_results(table: str):
        """Resume-integrity fallback: this cell's table from S3."""
        import io
        import boto3
        bucket = os.environ.get("REGISTRY_BUCKET")
        if not bucket:
            return None
        s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"))
        key = (f"eval_results/{args.benchmark}/{table}/"
               f"cell_id={args.run_id}.parquet")
        try:
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        except s3.exceptions.NoSuchKey:
            return None
        return pd.read_parquet(io.BytesIO(body))

    import pandas as pd
    scratch = Path(args.scratch_dir) if args.scratch_dir else None
    if not args.results_to_pvc and scratch is None:
        scratch = Path("/tmp/eval_scratch")  # S3-only mode needs a local write target
    try:
        runner = CachedRunner(
            cell, output_root=output_root,
            scoring_version=args.scoring_version,
            scratch_dir=scratch,
            prefetch_dir=(Path(args.prefetch_dir) / args.run_id
                          if args.prefetch_dir else None),
            prefetch_depth=args.prefetch_depth,
            copy_results=args.results_to_pvc,
            remote_results=None if args.no_registry else _remote_results,
        )
        summary = runner.run_once()
        log.info("done: processed=%d cached=%d forwards=%d",
                 summary["n_processed"], summary["n_cached"],
                 summary["n_forward_passes"])
    except Exception:
        if not args.no_registry:
            registry.register_eval_benchmark_done(
                **reg_kwargs, status="FAILED")
        raise

    # ── Sidecar: checkpoint step → tokens_seen ──────────────────────────
    results_root = (Path(summary["results_dir"])
                    if summary.get("results_dir") else None)
    if results_root is None and not args.results_to_pvc:
        # Fully cached, S3-only: everything is already on S3.
        log.info("✅ cell eval complete (fully cached): %s", args.run_id)
        if not args.no_registry:
            registry.register_eval_benchmark_done(
                **reg_kwargs, status="COMPLETE",
                parquet_path=f"s3://{os.environ.get('REGISTRY_BUCKET')}/"
                             f"eval_results/{args.benchmark}/")
        return
    meta_rows = []
    for step, path in ckpts:
        tokens_seen = None
        epoch = None
        meta_file = Path(path) / "metadata.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            tm = meta.get("token_metrics") or {}
            tokens_seen = (tm.get("estimated_tokens_at_step")
                           or tm.get("total_tokens_processed"))
            epoch = meta.get("epoch")
        meta_rows.append({
            "cell_id": args.run_id, "run_id": args.run_id,
            "architecture": arch, "lang": lang, "intervention": condition,
            "hp_rank": hp_rank, "seed": seed,
            "checkpoint_step": step, "tokens_seen": tokens_seen,
            "checkpoint_content_id": summary["checkpoint_ids"].get(step),
            "epoch": epoch,
            "benchmark": args.benchmark,
            "scoring_version": args.scoring_version,
            "stimuli_manifest_sha256": matched_manifest_sha,
            "stimuli_content_id": stimuli.stimuli_id,
            "stimuli_condition_output_sha256": json.dumps(
                (matched_manifest or {}).get("conditions", {})
                .get(condition, {}).get("output_sha256", {}), sort_keys=True),
        })
    ckpt_meta_dir = (results_root or output_root) / "checkpoints"
    ckpt_meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = ckpt_meta_dir / f"cell_id={args.run_id}.parquet"
    pd.DataFrame(meta_rows).to_parquet(meta_path, index=False)
    n_missing = sum(1 for r in meta_rows if r["tokens_seen"] is None)
    log.info("checkpoint sidecar → %s (%d rows, %d missing tokens_seen)",
             meta_path, len(meta_rows), n_missing)

    # ── Upload this run's parquets to S3 (laptop-pullable; non-fatal) ───
    s3_prefix = None
    try:
        import boto3
        bucket = os.environ.get("REGISTRY_BUCKET")
        if bucket:
            s3 = boto3.client("s3",
                              endpoint_url=os.environ.get("AWS_ENDPOINT_URL"))
            s3_prefix = f"eval_results/{args.benchmark}"
            for table in ("items", "pairs", "per_token", "checkpoints"):
                f = (results_root or output_root) / table \
                    / f"cell_id={args.run_id}.parquet"
                if f.exists():
                    s3.upload_file(str(f), bucket,
                                   f"{s3_prefix}/{table}/{f.name}")
            log.info("uploaded result parquets → s3://%s/%s/", bucket, s3_prefix)
    except Exception as exc:
        log.warning("S3 result upload failed (PVC copy remains): %s", exc)

    # ── Registry: benchmark done + metric summary ───────────────────────
    if not args.no_registry:
        metric_summary = None
        try:
            pairs_path = (results_root or output_root) / "pairs" \
                / f"cell_id={args.run_id}.parquet"
            if pairs_path.exists():
                pairs = pd.read_parquet(pairs_path)
                last = pairs[pairs.checkpoint_step == pairs.checkpoint_step.max()]
                metric_summary = {
                    "final_step": int(pairs.checkpoint_step.max()),
                    "n_checkpoints": int(pairs.checkpoint_step.nunique()),
                    "final_overt_pref": float(last.prefers_overt_meanlp.mean()),
                    "final_mean_logprob_diff":
                        float(last.log_prob_diff_overt_minus_null.mean()),
                }
        except Exception as exc:  # summary is best-effort, never fatal
            log.warning("metric summary failed: %s", exc)
        registry.register_eval_benchmark_done(
            **reg_kwargs, status="COMPLETE",
            parquet_path=str(output_root),
            metric_summary=metric_summary,
        )
    log.info("✅ cell eval complete: %s", args.run_id)


if __name__ == "__main__":
    main()
