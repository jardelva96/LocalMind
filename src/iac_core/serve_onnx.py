from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional, List, Union, Tuple

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def load_preprocess(path: Optional[str]):
    """Carrega mean/std/nomes de features, se existir."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    obj = json.loads(p.read_text(encoding="utf-8"))
    mean = np.array(obj.get("mean", []), dtype=np.float32) if "mean" in obj else None
    std = np.array(obj.get("std", []), dtype=np.float32) if "std" in obj else None
    names = obj.get("feature_names")
    return {"mean": mean, "std": std, "feature_names": names}


class PredictIn(BaseModel):
    # aceita [[...], [...]] ou lista plana [...]
    rows: Union[List[List[float]], List[float]]


class PredictOut(BaseModel):
    preds: List[int]
    probs: Optional[List[List[float]]] = None


def _to_2d_array(rows: Union[List[List[float]], List[float]]) -> np.ndarray:
    """Converte entrada para array 2D float32 (N, D)."""
    if isinstance(rows, list) and (len(rows) == 0 or isinstance(rows[0], (int, float))):
        return np.asarray([rows], dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _infer_input_dim(sess: ort.InferenceSession) -> Optional[int]:
    """Tenta inferir o input_dim pelo shape do primeiro input do modelo."""
    try:
        shape = sess.get_inputs()[0].shape  # tipicamente [None, D]
        if len(shape) >= 2 and isinstance(shape[1], (int, np.integer)):
            return int(shape[1])
    except Exception:
        pass
    return None


def build_app(session: ort.InferenceSession, input_name: str, input_dim: int, preprocess):
    app = FastAPI(title="IAZero ONNX Server", version="0.1.0")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "input": input_name, "dim": input_dim}

    @app.post("/predict", response_model=PredictOut)
    def predict(inp: PredictIn, return_proba: bool = True):
        try:
            X = _to_2d_array(inp.rows)
        except Exception:
            raise HTTPException(status_code=422, detail="Formato de 'rows' inválido. Envie [[...], ...] ou uma lista plana [...].")

        # checagem de dimensão
        if X.ndim != 2:
            raise HTTPException(status_code=422, detail=f"'rows' deve ser 2D (N,D). Recebido array com shape {list(X.shape)}.")

        if X.shape[1] > input_dim:
            X = X[:, :input_dim]  # corta extras
        elif X.shape[1] < input_dim:
            raise HTTPException(
                status_code=400,
                detail=f"Esperado {input_dim} features, recebido {X.shape[1]}. Complete/alimente com o número correto de colunas."
            )

        # normalização opcional
        if preprocess is not None and preprocess.get("mean") is not None and preprocess.get("std") is not None:
            mu, sd = preprocess["mean"], preprocess["std"]
            if mu.shape[0] >= input_dim and sd.shape[0] >= input_dim:
                sd_safe = np.where(sd == 0.0, 1.0, sd)
                X = (X - mu[:input_dim]) / sd_safe[:input_dim]

        # inferência
        try:
            logits = session.run(None, {input_name: X})[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Falha ao executar inferência ONNX: {e}")

        preds = np.argmax(logits, axis=1).astype(int)
        if return_proba:
            probs = softmax(logits).tolist()
            return {"preds": preds.tolist(), "probs": probs}
        return {"preds": preds.tolist(), "probs": None}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/export/iazero.onnx")
    ap.add_argument("--preprocess", default="models/export/preprocess.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--input-dim", type=int, default=None, help="Override do input_dim, caso o ONNX esteja dinâmico")
    ap.add_argument("--provider", default="CPUExecutionProvider", help="Sessão ONNXRuntime provider (padrão: CPUExecutionProvider)")
    args = ap.parse_args()

    sess = ort.InferenceSession(args.model, providers=[args.provider])
    input_name = sess.get_inputs()[0].name

    input_dim = args.input_dim or _infer_input_dim(sess)
    if input_dim is None:
        # fallback conservador
        input_dim = 32

    preprocess = load_preprocess(args.preprocess)
    app = build_app(sess, input_name, input_dim, preprocess)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
