from __future__ import annotations
import argparse
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

from iac_core.models.mlp import MLPClassifier

def dims_from_checkpoint(checkpoint: str) -> Tuple[Optional[int], Optional[int]]:
    try:
        state = torch.load(checkpoint, map_location="cpu")
    except FileNotFoundError:
        return None, None
    hp = {}
    if isinstance(state, dict):
        hp = state.get("hyper_parameters") or state.get("hparams", {}) or {}
    input_dim = hp.get("input_dim")
    num_classes = hp.get("num_classes")
    try:
        if input_dim is not None: input_dim = int(input_dim)
        if num_classes is not None: num_classes = int(num_classes)
    except Exception:
        pass
    return input_dim, num_classes

def load_model(checkpoint: Optional[str], input_dim: int, num_classes: int,
               hidden_dims=(128, 64), dropout=0.1, lr=1e-3, weight_decay=1e-4):
    model = MLPClassifier(input_dim=input_dim, hidden_dims=list(hidden_dims),
                          num_classes=num_classes, lr=lr, weight_decay=weight_decay, dropout=dropout)
    if checkpoint and Path(checkpoint).is_file():
        state = torch.load(checkpoint, map_location="cpu")
        if "state_dict" in state:
            model.load_state_dict(state["state_dict"], strict=False)
        else:
            model.load_state_dict(state, strict=False)
    model.eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="", help="caminho do .ckpt salvo (opcional)")
    ap.add_argument("--csv", type=str, required=True, help="CSV de entrada (features numéricas; se tiver 'target', será ignorado)")
    ap.add_argument("--out", type=str, default="predicoes.csv", help="arquivo de saída")
    ap.add_argument("--input-dim", type=int, default=None, help="número de features")
    ap.add_argument("--num-classes", type=int, default=None, help="número de classes")
    args = ap.parse_args()

    if not Path(args.csv).is_file():
        raise FileNotFoundError(f"CSV não encontrado: {args.csv}")

    # tenta descobrir dims do checkpoint, se existir
    ck_in_dim, ck_n_classes = (None, None)
    if args.checkpoint:
        ck_in_dim, ck_n_classes = dims_from_checkpoint(args.checkpoint)

    # carrega CSV e mantém só numéricas; se 'target' existir, dropa
    df = pd.read_csv(args.csv)
    num_df = df.select_dtypes(include=[np.number]).copy()
    if "target" in num_df.columns:
        num_df = num_df.drop(columns=["target"])
    X = num_df.to_numpy(dtype=np.float32)

    # decide input_dim
    in_dim = args.input_dim or ck_in_dim or X.shape[1]
    # se CSV tiver mais colunas do que o modelo espera, corta; se tiver menos, erro
    if X.shape[1] > in_dim:
        X = X[:, :in_dim]
    elif X.shape[1] < in_dim:
        raise SystemExit(f"CSV tem {X.shape[1]} features numéricas após limpeza, mas o modelo espera {in_dim}.")

    # decide num_classes
    n_classes = args.num_classes or ck_n_classes
    if n_classes is None:
        raise SystemExit("Informe --num-classes ou use --checkpoint de um treino para inferir automaticamente.")

    model = load_model(args.checkpoint or None, input_dim=int(in_dim), num_classes=int(n_classes))
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        preds = logits.argmax(dim=1).numpy()

    out = num_df.copy()
    out["pred"] = preds
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Salvo: {args.out}  (linhas: {len(out)})")

if __name__ == "__main__":
    main()