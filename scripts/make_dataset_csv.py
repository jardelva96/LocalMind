import argparse, numpy as np, pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/processed/dataset.csv")
ap.add_argument("--samples", type=int, default=1000)
ap.add_argument("--features", type=int, default=32)
ap.add_argument("--informative", type=int, default=16)
ap.add_argument("--classes", type=int, default=4)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

X, y = make_classification(
    n_samples=args.samples,
    n_features=args.features,
    n_informative=args.informative,
    n_redundant=0, n_repeated=0,
    n_classes=args.classes,
    random_state=args.seed
)
df = pd.DataFrame({f"f{i}": X[:, i] for i in range(X.shape[1])})
df["target"] = y
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(args.out, index=False)
print(f"Gerado: {args.out} shape: {df.shape}")