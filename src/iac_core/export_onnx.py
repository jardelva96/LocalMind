from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Tuple
import torch

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="models/export/model.onnx")
    ap.add_argument("--input-dim", type=int, default=None)
    ap.add_argument("--num-classes", type=int, default=None)
    args = ap.parse_args()

    ck_idim, ck_ncls = dims_from_checkpoint(args.checkpoint)
    idim = args.input_dim or ck_idim
    ncls = args.num_classes or ck_ncls
    if idim is None or ncls is None:
        raise SystemExit("Forneça --input-dim/--num-classes ou um checkpoint treinado com esses hyperparams.")

    model = MLPClassifier(input_dim=int(idim), hidden_dims=[128,64], num_classes=int(ncls), lr=1e-3, weight_decay=1e-4, dropout=0.1)
    state = torch.load(args.checkpoint, map_location="cpu")
    if "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()

    x = torch.randn(1, int(idim), dtype=torch.float32)
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, x, str(out_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17
    )
    print(f"ONNX salvo em: {out_path}")

if __name__ == "__main__":
    main()