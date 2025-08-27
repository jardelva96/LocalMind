from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/dataset.csv")
    ap.add_argument("--target-col", default="target")
    ap.add_argument("--out", default="models/export/preprocess.json")
    ap.add_argument("--input-dim", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    class_labels = None
    if args.target_col in df.columns:
        y = df[args.target_col]
        # mantém ordem estável; np.unique retorna ordenado
        if y.dtype.kind in "iu":
            class_labels = np.unique(y.values).tolist()
        else:
            class_labels = np.unique(y.values).astype(str).tolist()
        df = df.drop(columns=[args.target_col])

    num = df.select_dtypes(include=[np.number]).copy()
    if args.input_dim is not None and num.shape[1] > args.input_dim:
        num = num.iloc[:, :args.input_dim]

    mean = num.mean(axis=0).to_numpy(dtype=np.float32)
    std  = num.std(axis=0, ddof=0).replace(0.0, 1.0).to_numpy(dtype=np.float32)
    feature_names = list(num.columns)

    outp = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "feature_names": feature_names,
    }
    if class_labels is not None:
        outp["class_labels"] = class_labels

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(outp, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Salvo: {args.out}  (features: {len(feature_names)}, labels: {len(class_labels or [])})")

if __name__ == "__main__":
    main()
