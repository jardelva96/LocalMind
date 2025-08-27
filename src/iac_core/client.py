from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--endpoint", default="/predict")
    ap.add_argument("--flat", action="store_true", help="envia uma única linha plana (32 floats de 0.0..0.9 repetindo)")
    ap.add_argument("--proba", action="store_true", help="retorna probabilidades")
    ap.add_argument("--top-k", dest="top_k", type=int, default=0, help="retorna top-k {label, prob}")
    ap.add_argument("--pretty", action="store_true", help="imprime top-k em tabela amigável")
    ap.add_argument("--dim", type=int, default=32, help="dimensão sintética p/ --flat")
    ap.add_argument("--rows", type=int, default=1, help="linhas sintéticas p/ --flat (se >1, repete a primeira)")
    ap.add_argument("--csv", type=str, default="", help="arquivo CSV para amostrar linhas")
    ap.add_argument("--n", type=int, default=1, help="quantidade de linhas do CSV")
    ap.add_argument("--input-dim", type=int, default=None, help="limita primeiras N colunas numéricas do CSV")
    args = ap.parse_args()

    url = args.url.rstrip("/") + args.endpoint

    if args.csv:
        df = pd.read_csv(args.csv)
        num = df.select_dtypes(include=[np.number]).copy()
        if args.input_dim and num.shape[1] > args.input_dim:
            num = num.iloc[:, : args.input_dim]
        if args.n > len(num):
            raise SystemExit(f"--n ({args.n}) > linhas no CSV ({len(num)})")
        rows: List[List[float]] = num.iloc[: args.n].to_numpy(dtype=np.float32).tolist()
    elif args.flat:
        row = [float((i % 10) / 10.0) for i in range(args.dim)]
        rows = [row] if args.rows <= 1 else [row] * args.rows
    else:
        raise SystemExit("Use --flat ou --csv <arquivo>")

    params = {}
    if args.proba:
        params["return_proba"] = "true"
    if args.top_k and args.top_k > 0:
        params["top_k"] = str(args.top_k)

    payload = {"rows": rows if len(rows) > 1 else rows[0]}  # aceita lista plana também
    r = requests.post(url, params=params, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()

    if not args.pretty or not data.get("topk"):
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    # pretty print top-k
    topk = data["topk"]
    for i, items in enumerate(topk):
        print(f"\nLinha {i}:")
        print("{:<6} {:>8}".format("label", "prob%"))
        for it in items:
            label = it["label"]
            prob = float(it["prob"])
            print("{:<6} {:>7.1f}%".format(label, prob * 100.0))


if __name__ == "__main__":
    main()
