from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/predict")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target-col", default=None)
    ap.add_argument("--out", default="predicoes_out.csv")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--input-dim", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--proba", action="store_true")
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.target_col and args.target_col in df.columns:
        Xdf = df.drop(columns=[args.target_col])
    else:
        Xdf = df.copy()
    Xdf = Xdf.select_dtypes(include=[np.number])

    if args.input_dim and Xdf.shape[1] > args.input_dim:
        Xdf = Xdf.iloc[:, : args.input_dim]

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["x-api-key"] = args.api_key

    preds, topk_all = [], []
    for i in range(0, len(Xdf), args.batch):
        chunk = Xdf.iloc[i : i + args.batch].to_numpy(dtype=np.float32).tolist()
        resp = requests.post(
            args.url,
            params={"return_proba": args.proba, "top_k": args.top_k},
            headers=headers,
            data=json.dumps({"rows": chunk}),
        )
        resp.raise_for_status()
        data = resp.json()
        preds.extend(data["preds"])
        if args.top_k:
            topk_all.extend(data.get("topk", [[] for _ in range(len(chunk))]))

    out = df.copy()
    out["pred"] = preds
    if args.top_k:
        out["topk_json"] = [json.dumps(x, ensure_ascii=False) for x in topk_all]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Salvo: {args.out}  (linhas: {len(out)})")


if __name__ == "__main__":
    main()
