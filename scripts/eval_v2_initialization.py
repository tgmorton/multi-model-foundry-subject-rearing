#!/usr/bin/env python3
"""Generate and evaluate exact pre-training initialization states.

This is deliberately separate from the training checkpoint scheduler.  The
trainer's ``checkpoint-0`` is saved after the first optimizer update; this
script reconstructs the state immediately before that update from the
run's architecture/config/tokenizer and random seed.

One process handles one (architecture, seed) tuple.  The same initialization
state is evaluated for every existing cell that shares that tuple (different
conditions and HP ranks do not change the initialized weights).  State and
results are written atomically to S3, with a local PVC copy retained for
audit/recovery.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml


log = logging.getLogger("eval_v2_initialization")
DEFAULT_BENCHMARK = "null_subj_v2_init"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def state_sha256(model) -> str:
    """Stable digest of the model state independent of key insertion order."""
    import torch

    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        t = value.detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(t.dtype).encode("ascii"))
        h.update(repr(tuple(t.shape)).encode("ascii"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def load_cells(path: Path, arch: str, seed: int) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for row in rows:
            row["seed"] = int(row["seed"])
            row["hp_rank"] = int(row["hp_rank"])
    else:
        payload = json.loads(path.read_text())
        rows = payload.get("runs", []) if isinstance(payload, dict) else payload
        for row in rows:
            row.setdefault("cell_id", row.get("run_id"))
            row.setdefault("intervention", row.get("condition"))
    out = [r for r in rows
           if r["architecture"] == arch and int(r["seed"]) == int(seed)]
    if not out:
        raise SystemExit(f"no cells for arch={arch} seed={seed} in {path}")
    return out


def tokenizer_dir(arch: str, lang: str) -> str:
    return f"{lang}_bert_wordpiece" if arch == "bert_large" else f"{lang}_shared_unigram"


class HFEvalTokenizer:
    """Small adapter used by the v2 stimulus cache for BERT."""

    def __init__(self, hf):
        self.hf = hf

    def encode(self, text: str):
        return self.hf.encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self.hf.decode(list(ids), skip_special_tokens=False)

    def bos_id(self):
        return getattr(self.hf, "cls_token_id", None)

    def eos_id(self):
        return getattr(self.hf, "sep_token_id", None)

    def get_piece_size(self):
        return len(self.hf)


def build_cfg(repo: Path, arch: str, lang: str, tok_dir: Path):
    from model_foundry.cli import load_config
    from model_foundry.training.tokenization import load_tokenizer

    cfg_path = repo / "configs" / "sweeps" / "baselines" / f"{arch}_{lang}.yaml"
    cfg = load_config(str(cfg_path))
    train_tok = load_tokenizer(str(tok_dir))
    actual_vocab = len(train_tok)
    if cfg.tokenizer.vocab_size != actual_vocab:
        log.info("vocab sync %d -> %d", cfg.tokenizer.vocab_size, actual_vocab)
        cfg.tokenizer.vocab_size = actual_vocab
    return cfg, train_tok, cfg_path


def build_stimuli(repo: Path, arch: str, lang: str, train_tok, tok_dir: Path,
                  cache_root: Path, condition: Optional[str] = None,
                  matched_stimuli_root: Optional[Path] = None):
    from evaluation.stimuli_cache import build_or_load_cache

    if matched_stimuli_root is not None:
        stimuli_dir = matched_stimuli_root / str(condition) / lang
    else:
        stimuli_dir = (repo / "evaluation" / "stimuli" /
                       "null-subj-v2" / "staging" / lang)
    stimuli_paths = sorted(stimuli_dir.glob("*.csv"))
    if not stimuli_paths:
        raise SystemExit(f"no stimuli under {stimuli_paths}")
    if arch == "bert_large":
        cache_tok = HFEvalTokenizer(train_tok)
        hash_file = next((p for p in (tok_dir / "tokenizer.json",
                                      tok_dir / "vocab.txt",
                                      tok_dir / "tokenizer.model") if p.exists()), None)
        if hash_file is None:
            raise SystemExit(f"no BERT tokenizer file in {tok_dir}")
        pad_id = train_tok.pad_token_id
    else:
        import sentencepiece as spm
        hash_file = tok_dir / "tokenizer.model"
        sp = spm.SentencePieceProcessor()
        sp.load(str(hash_file))
        cache_tok = sp
        pad_id = sp.pad_id() if sp.pad_id() >= 0 else max(sp.unk_id(), 0)
    stimuli = build_or_load_cache(stimuli_paths, cache_tok, hash_file,
                                  cache_root / arch)
    return stimuli, cache_tok, hash_file, pad_id


def build_model_factory(repo: Path, arch: str, lang: str, cfg,
                        actual_vocab: int):
    from model_foundry.model import create_model

    def factory():
        # The trainer uses the same model factory.  Eager attention changes
        # eval speed only; it does not alter initialization weights.
        if arch in ("gpt2_small", "gpt2_medium", "gpt2_large", "bert_large"):
            return create_model(cfg, attn_implementation="eager")
        return create_model(cfg)

    return factory


def load_unigrams(s3, bucket: str, conditions: Iterable[str], work: Path):
    from evaluation.unigram import UnigramTable

    out = {}
    work.mkdir(parents=True, exist_ok=True)
    for condition in sorted(set(conditions)):
        key = f"unigrams/en_{condition}__shared_unigram.pkl"
        local = work / Path(key).name
        if not local.exists():
            s3.download_file(bucket, key, str(local))
        out[condition] = UnigramTable.load(local)
    return out


def upload_once(s3, bucket: str, path: Path, key: str, metadata: dict[str, str]):
    """Upload only if absent; refuse a same-key different-content collision."""
    digest = sha256_file(path)
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
        prior = (head.get("Metadata") or {}).get("sha256")
        if prior == digest:
            return "cached"
        raise RuntimeError(f"S3 collision at s3://{bucket}/{key}: sha256 differs")
    except Exception as exc:
        code = getattr(getattr(exc, "response", None), "get", lambda *_: None)("Error", {}).get("Code")
        if code not in ("404", "NoSuchKey", "NotFound"):
            # botocore's ClientError exposes response; preserve non-404 errors.
            if hasattr(exc, "response"):
                raise
    md = dict(metadata)
    md["sha256"] = digest
    s3.upload_file(str(path), bucket, key, ExtraArgs={"Metadata": md})
    return "uploaded"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=Path("/opt/repo"))
    ap.add_argument("--data-root", type=Path, default=Path("/mnt/data"))
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--bucket", default=os.environ.get("REGISTRY_BUCKET", "thomas-subject-drop-artifacts"))
    ap.add_argument("--endpoint", default=os.environ.get("AWS_ENDPOINT_URL"))
    ap.add_argument("--image-digest", default=os.environ.get("IMAGE_DIGEST", "unknown"))
    ap.add_argument("--code-ref", default=os.environ.get("GIT_REF", "unknown"))
    ap.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    ap.add_argument(
        "--matched-stimuli-root", type=Path,
        help="Condition-matched root: <root>/<condition>/<lang>/*.csv",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.arch not in {"bert_large", "gpt2_small", "gpt2_medium",
                         "gpt2_large", "lstm", "mamba_370m"}:
        raise SystemExit(f"unsupported architecture {args.arch}")

    import boto3
    import torch
    from model_foundry.utils import set_seed
    from evaluation.runners.per_model_runner import CellSpec, PerModelRunner, _build_pair_results
    from evaluation.output_v2 import write_cell_results
    from evaluation.core.pll_surprisal import make_pll_scorer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    cells = load_cells(args.cells, args.arch, args.seed)
    lang = "en"
    data_root = args.data_root
    tok_path = data_root / "tokenizers" / tokenizer_dir(args.arch, lang)
    cfg, train_tok, cfg_path = build_cfg(args.repo, args.arch, lang, tok_path)
    tok_hash_path = next((p for p in (
        tok_path / "tokenizer.json", tok_path / "vocab.txt",
        tok_path / "tokenizer.model") if p.exists()), None)
    if tok_hash_path is None:
        raise SystemExit(f"no hashable tokenizer file in {tok_path}")
    actual_vocab = len(train_tok)
    factory = build_model_factory(args.repo, args.arch, lang, cfg, actual_vocab)

    # Re-seed immediately before model construction, matching Trainer._train_loop.
    set_seed(args.seed)
    model = factory()
    state_hash = state_sha256(model)

    init_dir = (data_root / "models" / "initialization" / args.arch /
                f"seed-{args.seed}" / "checkpoint_-1")
    init_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(init_dir))
    metadata = {
        "schema_version": "foundry-init-v1",
        "checkpoint_step": -1,
        "tokens_seen": 0,
        "epoch": 0,
        "pre_training_initialization": True,
        "architecture": args.arch,
        "language": lang,
        "seed": args.seed,
        "code_ref": args.code_ref,
        "image_digest": args.image_digest,
        "baseline_config": str(cfg_path),
        "baseline_config_sha256": sha256_file(cfg_path),
        "tokenizer_file": str(tok_hash_path),
        "tokenizer_file_sha256": sha256_file(tok_hash_path),
        "actual_vocab_size": actual_vocab,
        "model_state_sha256": state_hash,
        "shared_across_cells": True,
        "cell_ids": [r["cell_id"] for r in cells],
    }
    (init_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    s3 = boto3.client("s3", endpoint_url=args.endpoint,
                      region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-1"))
    init_prefix = f"initialization/{args.arch}/seed-{args.seed}/checkpoint_-1"
    common_md = {"schema": "foundry-init-v1", "code_ref": args.code_ref,
                 "image_digest": args.image_digest}
    for p in sorted(init_dir.iterdir()):
        # A prior run may have materialized the identical deterministic model
        # state while recording a narrower cell list.  Preserve that durable
        # metadata rather than treating the changed fan-out list as an
        # artifact collision; any state-hash mismatch remains fatal.
        if p.name == "metadata.json":
            key = f"{init_prefix}/{p.name}"
            try:
                prior = json.loads(s3.get_object(
                    Bucket=args.bucket, Key=key)["Body"].read())
            except Exception as exc:
                code = (getattr(exc, "response", {}) or {}).get(
                    "Error", {}).get("Code")
                if code not in ("404", "NoSuchKey", "NotFound"):
                    raise
            else:
                local = json.loads(p.read_text())
                if (prior.get("model_state_sha256") ==
                        local.get("model_state_sha256")):
                    log.info("reusing compatible initialization metadata %s",
                             key)
                    continue
                raise RuntimeError(
                    f"S3 collision at s3://{args.bucket}/{key}: "
                    "model_state_sha256 differs")
        if p.is_file():
            upload_once(s3, args.bucket, p, f"{init_prefix}/{p.name}", common_md)

    # Move the model to the selected device once, and reuse it for all cells.
    model = model.to(args.device)
    model.eval()
    scorer = None
    if args.arch == "bert_large":
        if train_tok.mask_token_id is None or train_tok.sep_token_id is None:
            raise SystemExit("BERT tokenizer missing mask/sep ids")
        scorer = make_pll_scorer(mask_id=train_tok.mask_token_id,
                                 sep_id=train_tok.sep_token_id)

    # Evaluate the shared initialization once per distinct condition-specific
    # stimulus set, then fan those scores out across HP-rank cells.  We cannot
    # reuse one generic base_items table in the matched regime because the
    # stimulus content differs by intervention.
    base_items_by_condition = {}
    for condition in sorted({r["intervention"] for r in cells}):
        stimuli, cache_tok, tok_hash_path, pad_id = build_stimuli(
            args.repo, args.arch, lang, train_tok, tok_path,
            data_root / "eval_v2" / args.benchmark / "stimuli_cache" / condition,
            condition=condition,
            matched_stimuli_root=args.matched_stimuli_root,
        )
        representative = next(r for r in cells if r["intervention"] == condition)
        base_cell = CellSpec(
            cell_id=representative["cell_id"], architecture=args.arch,
            intervention=condition, rep=args.seed,
            checkpoint_root=init_dir, model_factory=lambda: model,
            stimuli=stimuli, unigram=None, device=args.device, batch_size=64,
            scorer=scorer, pad_id=pad_id,
        )
        runner = PerModelRunner(base_cell)
        # `model` is the exact untouched state constructed immediately above.
        # Do not round-trip it through PerModelRunner.load_checkpoint here.
        base_items_by_condition[condition], _ = runner.evaluate_checkpoint(
            -1, init_dir)
    s3_ckpt = f"s3://{args.bucket}/{init_prefix}"

    unigrams = {}
    if args.arch != "bert_large":
        unigrams = load_unigrams(s3, args.bucket,
                                 [r["intervention"] for r in cells],
                                 data_root / "tmp" / "foundry-init-unigrams")

    output_root = data_root / "eval_v2" / args.benchmark
    for cell in cells:
        condition = cell["intervention"]
        items = []
        uni = unigrams.get(condition)
        for row in base_items_by_condition[condition]:
            target_uni = None
            slor = None
            if uni is not None and row.target_n_tokens:
                target_uni = float(uni.sum_log_prob(row.per_token_ids))
                slor = ((row.target_sum_log_prob - target_uni) /
                        row.target_n_tokens)
            items.append(replace(
                row, cell_id=cell["cell_id"], intervention=condition,
                rep=args.seed, checkpoint_path=s3_ckpt,
                target_unigram_sum_log_prob=target_uni, slor=slor,
            ))
        pairs = _build_pair_results(items)
        cell_out = output_root
        write_cell_results(cell_out, cell["cell_id"], items, pairs,
                           include_per_token=True)
        for table in ("items", "pairs", "per_token"):
            p = cell_out / table / f"cell_id={cell['cell_id']}.parquet"
            key = f"eval_results/{args.benchmark}/{table}/{p.name}"
            upload_once(s3, args.bucket, p, key, common_md)
        side = cell_out / "checkpoints" / f"cell_id={cell['cell_id']}.parquet"
        side.parent.mkdir(parents=True, exist_ok=True)
        import pandas as pd
        pd.DataFrame([{
            "cell_id": cell["cell_id"], "architecture": args.arch,
            "intervention": condition, "hp_rank": int(cell["hp_rank"]),
            "seed": args.seed, "checkpoint_step": -1,
            "tokens_seen": 0, "epoch": 0,
            "initialization_state": s3_ckpt,
            "model_state_sha256": state_hash,
            "code_ref": args.code_ref, "image_digest": args.image_digest,
        }]).to_parquet(side, index=False)
        upload_once(s3, args.bucket, side,
                    f"eval_results/{args.benchmark}/checkpoints/{side.name}",
                    common_md)
        log.info("completed %s (%d item, %d pair rows)",
                 cell["cell_id"], len(items), len(pairs))

    # Per-seed provenance record is itself an atomic S3 object and serves as
    # the durable index for this process. It intentionally does not mutate the
    # existing production run_registry records.
    record = dict(metadata)
    record["s3_initialization_prefix"] = f"s3://{args.bucket}/{init_prefix}/"
    record["s3_eval_prefix"] = f"s3://{args.bucket}/eval_results/{args.benchmark}/"
    record["n_cells"] = len(cells)
    rec_path = init_dir.parent / "initialization_record.json"
    rec_path.write_text(json.dumps(record, indent=2) + "\n")
    upload_once(s3, args.bucket, rec_path,
                f"initialization/{args.arch}/seed-{args.seed}/initialization_record.json",
                common_md)
    log.info("INIT COMPLETE arch=%s seed=%d state_sha256=%s cells=%d",
             args.arch, args.seed, state_hash, len(cells))


if __name__ == "__main__":
    main()
