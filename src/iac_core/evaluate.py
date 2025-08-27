from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from iac_core.models.mlp import MLPClassifier

def dims_from_checkpoint(checkpoint: str) -> Tuple[Optional[int], Optional[int]]:
    try:
        state = torch.load(checkpoint, map_location="cpu")
    except FileNotFoundError:
        return None, None
    hp = {}
    if isinstance(state, dict):
        hp = state.get("hyper_parameters") or state.get("hparams", {}) or {}
    idim = hp.get("input_dim"); ncls = hp.get("num_classes")
    try:
        if idim is not None: idim = int(idim)
        if ncls is not None: ncls = int(ncls)
    except Exception:
        pass
    return idim, ncls

def load_model(ckpt: str, input_dim: int, num_classes: int):
    m = MLPClassifier(input_dim=input_dim, hidden_dims=[128,64], num_classes=num_classes, lr=1e-3, weight_decay=1e-4, dropout=0.1)
    if ckpt and Path(ckpt).is_file():
        state = torch.load(ckpt, map_location="cpu")
        if "state_dict" in state:
            m.load_state_dict(state["state_dict"], strict=False)
        else:
            m.load_state_dict(state, strict=False)
    m.eval()
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="caminho do .ckpt")
    ap.add_argument("--csv", required=True, help="CSV com features + coluna alvo")
    ap.add_argument("--target-col", dest="target_col", default="target")
    ap.add_argument("--outdir", default="logs/eval")
    ap.add_argument("--input-dim", dest="input_dim", type=int, default=None)
    ap.add_argument("--num-classes", dest="num_classes", type=int, default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    tcol = args.target_col
    if tcol not in df.columns:
        raise SystemExit(f"Coluna alvo '{tcol}' não encontrada.")

    # separa alvo e features numéricas
    y_raw = df[tcol].to_numpy()
    Xdf = df.drop(columns=[tcol]).select_dtypes(include=[np.number]).copy()

    # codificação do alvo (texto -> int) compatível com treino (np.unique ordena)
    if y_raw.dtype.kind in "iu":
        classes = np.unique(y_raw)
        y_true = y_raw.astype(int)
    else:
        classes, y_true = np.unique(y_raw, return_inverse=True)
    n_cls_from_data = len(classes)

    # inferência das dims
    ck_in_dim, ck_ncls = dims_from_checkpoint(args.checkpoint)
    in_dim = args.input_dim or ck_in_dim or Xdf.shape[1]
    n_classes = args.num_classes or ck_ncls or n_cls_from_data

    X = Xdf.to_numpy(dtype=np.float32)
    if X.shape[1] > in_dim:   # corta extras
        X = X[:, :in_dim]
        Xdf = Xdf.iloc[:, :in_dim]
    elif X.shape[1] < in_dim:
        raise SystemExit(f"CSV tem {X.shape[1]} features numéricas; o modelo espera {in_dim}.")

    model = load_model(args.checkpoint, input_dim=int(in_dim), num_classes=int(n_classes))
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        y_pred = logits.argmax(dim=1).cpu().numpy()

    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))

    # salva métricas
    metrics = {"accuracy": acc, "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f1)}
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # salva CSV com y_true e y_pred
    out_csv = Xdf.copy()
    out_csv["y_true"] = y_true
    out_csv["y_pred"] = y_pred
    out_csv.to_csv(outdir / "predicoes_com_rotulo.csv", index=False)

    # matriz de confusão (PNG)
    fig = plt.figure(figsize=(5,4))
    im = plt.imshow(cm, interpolation="nearest")  # colormap default
    plt.title("Matriz de confusão")
    plt.colorbar(im)
    ticks = np.arange(n_classes)
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.xlabel("Predito"); plt.ylabel("Verdadeiro")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    fig.savefig(outdir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print("Métricas:", metrics)
    print("Arquivos salvos em:", outdir)

if __name__ == "__main__":
    main()