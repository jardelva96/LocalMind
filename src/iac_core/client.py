from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import requests


def make_rows_from_csv(path: str, target_col: str = "target",
                       n: Optional[int] = None, input_dim: Optional[int] = None) -> List[List[float]]:
    df = pd.read_csv(path)
    if target_col in df.columns:
        df = df.drop(columns=[target_col])
    num = df.select_dtypes(include=[np.number]).copy()
    if input_dim and num.shape[1] > input_dim:
        num = num.iloc[:, :input_dim]
    if n:
        num = num.iloc[:n]
    return num.to_numpy(dtype=np.float32).tolist()


def make_dummy_rows(dim: int = 32, rows: int = 1) -> List[List[float]]:
    out: List[List[float]] = []
    for r in range(rows):
        out.append([float((i % 10) / (10.0 + r)) for i in range(dim)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL do servidor FastAPI")
    ap.add_argument("--endpoint", default="/predict")
    ap.add_argument("--flat", action="store_true", help="envia lista plana em vez de [[...]]")
    ap.add_argument("--proba", action="store_true", help="solicita probabilidades (return_proba=true)")

    # geração de dados de exemplo
    ap.add_argument("--dim", type=int, default=32, help="número de features para dados sintéticos")
    ap.add_argument("--rows", type=int, default=1, help="quantas linhas sintéticas gerar")

    # leitura de CSV (opcional)
    ap.add_argument("--csv", type=str, default="", help="CSV para enviar (usa apenas colunas numéricas; 'target' é ignorada)")
    ap.add_argument("--n", type=int, default=None, help="quantas linhas do CSV usar")
    ap.add_argument("--input-dim", type=int, default=None, help="corta colunas extras ao ler o CSV")

    args = ap.parse_args()

    if args.csv:
        rows = make_rows_from_csv(args.csv, n=args.n, input_dim=args.input_dim)
    else:
        rows = make_dummy_rows(dim=args.dim, rows=args.rows)

    payload = {"rows": rows[0] if (args.flat and len(rows) > 0) else rows}
    url = args.url.rstrip("/") + args.endpoint
    params = {"return_proba": "true" if args.proba else "false"}

    r = requests.post(url, json=payload, params=params, timeout=15)
    r.raise_for_status()
    print(json.dumps(r.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
