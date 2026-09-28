"""Compare fp32 vs fp16 CPU evals of the same checkpoints (fp16 science
check), plus the June GPU eval as a CPU-vs-GPU noise reference."""
import io
import boto3
import numpy as np
import pandas as pd

S = boto3.Session(profile_name="nrp").client(
    "s3", endpoint_url="https://s3-west.nrp-nautilus.io")
B = "thomas-subject-drop-artifacts"
COLS = ["checkpoint_step", "category", "condition", "item_id",
        "overt_mean_log_prob", "null_mean_log_prob", "prefers_overt_meanlp"]
KEY = ["checkpoint_step", "category", "condition", "item_id"]


def get(bench, rid):
    k = (f"fp16_verify_scratch/{rid}/pairs/cell_id={rid}.parquet" if bench == "scratch"
         else f"eval_results/{bench}/pairs/cell_id={rid}.parquet")
    return pd.read_parquet(io.BytesIO(S.get_object(Bucket=B, Key=k)["Body"].read()),
                           columns=COLS)


def compare(a, b, la, lb):
    j = a.merge(b, on=KEY, suffixes=("_a", "_b"))
    d_o = (j.overt_mean_log_prob_a - j.overt_mean_log_prob_b).abs()
    d_n = (j.null_mean_log_prob_a - j.null_mean_log_prob_b).abs()
    flip = (j.prefers_overt_meanlp_a != j.prefers_overt_meanlp_b)
    ra = j.groupby("checkpoint_step").prefers_overt_meanlp_a.mean()
    rb = j.groupby("checkpoint_step").prefers_overt_meanlp_b.mean()
    print(f"\n{la} vs {lb}: {len(j):,} paired items over "
          f"{j.checkpoint_step.nunique()} checkpoints")
    print(f"  per-token logprob |Δ|: max {max(d_o.max(), d_n.max()):.4f}  "
          f"median {np.median(np.r_[d_o, d_n]):.5f}")
    print(f"  items whose overt/null preference FLIPS: {flip.sum()} "
          f"({flip.mean():.3%})")
    print(f"  per-checkpoint preference rate |Δ|: max {(ra - rb).abs().max():.4f}")
    return flip.mean(), (ra - rb).abs().max()


fp32 = get("scratch", "gpt2_large-en-baseline-h0-s9990001")
fp16 = get("scratch", "gpt2_large-en-baseline-h0-s9990002")
june = get("null_subj_v2_condition_matched_v1", "gpt2_large-en-baseline-h0-s137")
june = june[june.checkpoint_step.isin(fp32.checkpoint_step.unique())]
f16_flip, f16_rate = compare(fp32, fp16, "fp32-CPU", "fp16-CPU")
ref_flip, ref_rate = compare(fp32, june, "fp32-CPU", "June-GPU")
print(f"\nVERDICT: fp16 flips {f16_flip:.3%} of preferences vs "
      f"{ref_flip:.3%} from CPU-vs-GPU alone")
print("  -> fp16 effect is " + ("WITHIN existing numerical noise — safe"
      if f16_flip <= max(ref_flip, 0.005) else "LARGER than existing noise — review"))
