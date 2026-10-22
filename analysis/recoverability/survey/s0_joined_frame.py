"""Step 0: (re)build the joined rater frame, one row per selected instance.

Source (S3, bucket thomas-subject-drop-artifacts, profile 'nrp'):
  recoverability/analysis/selection_v5/{robbi,bert,gpt2m}/train_90M/<genre>.parquet
  columns line_idx, token_i, form, rank_value (= surprisal, nats),
          info_decile (= decile, 0 = most recoverable), rand_decile
Output columns: genre, line_idx, token_i, form, d_robbi, s_robbi, rand_decile,
                d_bert, s_bert, d_gpt2m, s_gpt2m   (robbi's rand_decile; it is
                identical across labels by construction, checked here)

--source s3            download the keys into --cache (then build)
--source DIR           a local mirror laid out by S3 key (DIR/<key>)
Writes <out>/robbi_joined.parquet and <out>/joined_sources.json (keys, sizes,
md5 of local copies and, with --head, the S3 ETags for comparison).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import survey_common as C  # noqa: E402

K = ["genre", "line_idx", "token_i"]


def s3_client():
    import boto3
    return boto3.Session(profile_name=C.S3_PROFILE).client(
        "s3", endpoint_url=C.S3_ENDPOINT)


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_label(root: Path, lab: str) -> pd.DataFrame:
    fr = []
    for g in C.GENRES:
        d = pq.read_table(root / C.SEL_PREFIX / lab / "train_90M" / f"{g}.parquet"
                          ).to_pandas()
        d.insert(0, "genre", g)
        fr.append(d)
    return pd.concat(fr, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--source", required=True, help="'s3' or a local mirror dir")
    ap.add_argument("--cache", type=Path, default=None,
                    help="download dir for --source s3 (default <out>/_s3cache)")
    ap.add_argument("--head", action="store_true",
                    help="also HEAD each key on S3 and record ETag/size")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or C.default_paths(a.data_root)["out"]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    keys = C.selection_keys()
    if a.source == "s3":
        root = a.cache or (out / "_s3cache")
        cli = s3_client()
        for k in keys:
            dst = root / k
            dst.parent.mkdir(parents=True, exist_ok=True)
            cli.download_file(C.S3_BUCKET, k, str(dst))
    else:
        root = Path(a.source)
    rec = []
    cli = s3_client() if a.head else None
    for k in keys:
        p = root / k
        r = {"key": k, "local_size": p.stat().st_size, "local_md5": md5(p)}
        if cli is not None:
            h = cli.head_object(Bucket=C.S3_BUCKET, Key=k)
            r.update(s3_size=h["ContentLength"], s3_etag=h["ETag"].strip('"'),
                     s3_last_modified=str(h["LastModified"]))
            # multipart ETags ("<md5-of-md5s>-N") are not a content md5
            r["md5_matches_etag"] = (None if "-" in r["s3_etag"]
                                     else r["s3_etag"] == r["local_md5"])
            r["size_matches"] = r["s3_size"] == r["local_size"]
        rec.append(r)

    rb = load_label(root, "robbi").rename(
        columns={"info_decile": "d_robbi", "rank_value": "s_robbi"})
    m = rb[K + ["form", "d_robbi", "s_robbi", "rand_decile"]]
    checks = {}
    for lab in ("bert", "gpt2m"):
        d = load_label(root, lab)
        x = m[K + ["rand_decile"]].merge(d[K + ["rand_decile"]], on=K, how="left",
                                         suffixes=("", "_o"))
        checks[f"rand_identical_{lab}"] = float((x.rand_decile == x.rand_decile_o).mean())
        m = m.merge(d[K + ["info_decile", "rank_value"]].rename(
            columns={"info_decile": f"d_{lab}", "rank_value": f"s_{lab}"}),
            on=K, how="left")
        checks[f"coverage_{lab}"] = float(m[f"d_{lab}"].notna().mean())
    m.to_parquet(out / "robbi_joined.parquet", index=False)
    info = {"bucket": C.S3_BUCKET, "endpoint": C.S3_ENDPOINT, "source": str(a.source),
            "keys": rec, "n_rows": int(len(m)), "checks": checks,
            "seconds": round(time.time() - t0, 1)}
    (out / "joined_sources.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "keys"}, indent=2))


if __name__ == "__main__":
    main()
