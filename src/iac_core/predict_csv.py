from __future__ import annotations
import argparse, json, time
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import requests

def try_get_healthz(url: str, headers: dict, timeout: int = 15) -> dict | None:
    base = url.rsplit("/predict", 1)[0]
    health = f"{base}/healthz" if base else url
    try:
        r = requests.get(health, headers=headers, timeout=timeout)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None

def reorder_like(df: pd.DataFrame, names: List[str]) -> pd.DataFrame:
    for c in names:
        if c not in df.columns:
            df[c] = 0.0
    return df[names]

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
    ap.add_argument("--preprocess", default=None, help="Caminho do preprocess.json para alinhar colunas")
    ap.add_argument("--fillna", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    Xdf = df.drop(columns=[args.target_col]) if (args.target_col and args.target_col in df.columns) else df.copy()
    Xdf = Xdf.select_dtypes(include=[np.number]).copy()

    feature_names: Optional[List[str]] = None
    if args.preprocess:
        try:
            pp = json.loads(Path(args.preprocess).read_text(encoding="utf-8"))
            if isinstance(pp.get("feature_names"), list):
                feature_names = [str(x) for x in pp["feature_names"]]
        except Exception:
            feature_names = None
    if feature_names is None:
        headers_health = {"x-api-key": args.api_key} if args.api_key else {}
        health = try_get_healthz(args.url, headers=headers_health, timeout=args.timeout)
        if isinstance(health, dict) and isinstance(health.get("feature_names"), list):
            feature_names = [str(x) for x in health["feature_names"]]

    if feature_names:
        Xdf = reorder_like(Xdf, feature_names)
    elif args.input_dim and Xdf.shape[1] > args.input_dim:
        Xdf = Xdf.iloc[:, : args.input_dim]

    if args.fillna is not None:
        Xdf = Xdf.fillna(float(args.fillna))

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["x-api-key"] = args.api_key

    want_proba = bool(args.proba or args.top_k > 0)

    preds: list[int] = []
    pred_labels: list[str] = []
    topk_all: list[list[dict]] = []

    for i in range(0, len(Xdf), args.batch):
        chunk_df = Xdf.iloc[i : i + args.batch]
        rows = chunk_df.to_numpy(dtype=np.float32).tolist()

        attempt = 0
        while True:
            try:
                resp = requests.post(
                    args.url,
                    params={"return_proba": want_proba, "top_k": int(args.top_k)},
                    headers=headers,
                    json={"rows": rows},
                    timeout=args.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception:
                attempt += 1
                if attempt > max(1, args.retries):
                    raise
                time.sleep(min(2 * attempt, 5))

        preds.extend(data.get("preds", []))
        if data.get("pred_labels"):
            pred_labels.extend(data["pred_labels"])
        if args.top_k:
            tlist = data.get("topk") or [[] for _ in range(len(rows))]
            topk_all.extend(tlist)

    out = df.copy()
    out["pred"] = preds
    if pred_labels:
        out["pred_label"] = pred_labels
    if args.top_k:
        top1_label, top1_prob = [], []
        for items in topk_all:
            if items:
                top1_label.append(items[0].get("label"))
                top1_prob.append(float(items[0].get("prob", 0.0)))
            else:
                top1_label.append(None)
                top1_prob.append(None)
        out["top1_label"] = top1_label
        out["top1_prob"] = top1_prob
        out["topk_json"] = [json.dumps(x, ensure_ascii=False) for x in topk_all]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Salvo: {args.out}  (linhas: {len(out)})")

if __name__ == "__main__":
    main()