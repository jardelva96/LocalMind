from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Optional, List, Union
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)

def load_preprocess(path: Optional[str]):
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    obj = json.loads(p.read_text(encoding="utf-8"))
    mean = np.array(obj.get("mean", []), dtype=np.float32) if "mean" in obj else None
    std  = np.array(obj.get("std",  []), dtype=np.float32) if "std"  in obj else None
    names = obj.get("feature_names")
    return {"mean": mean, "std": std, "feature_names": names}

class PredictIn(BaseModel):
    rows: List[List[float]]

class PredictOut(BaseModel):
    preds: List[int]
    probs: Optional[List[List[float]]] = None

def build_app(session: ort.InferenceSession, input_name: str, input_dim: int, preprocess):
    app = FastAPI(title="IAZero ONNX Server", version="0.1.0")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "input": input_name, "dim": input_dim}

    @app.post("/predict", response_model=PredictOut)
    def predict(inp: PredictIn, return_proba: bool = True):
        X = np.array(inp.rows, dtype=np.float32)
        # ajusta dimensões
        if X.shape[1] > input_dim:
            X = X[:, :input_dim]
        elif X.shape[1] < input_dim:
            raise ValueError(f"Esperado {input_dim} features, recebido {X.shape[1]}.")

        # normalização opcional
        if preprocess is not None and preprocess.get("mean") is not None and preprocess.get("std") is not None:
            mu, sd = preprocess["mean"], preprocess["std"]
            if mu.shape[0] >= input_dim and sd.shape[0] >= input_dim:
                sd_safe = np.where(sd == 0.0, 1.0, sd)
                X = (X - mu[:input_dim]) / sd_safe[:input_dim]

        ort_inputs = {input_name: X}
        logits = session.run(None, ort_inputs)[0]
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
    args = ap.parse_args()

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    # tenta deduzir input_dim pela forma do primeiro input
    shape = sess.get_inputs()[0].shape
    # shape típico: ["batch", input_dim]
    input_dim = int(shape[1]) if (len(shape) >= 2 and isinstance(shape[1], (int, np.integer))) else None
    if input_dim is None:
        # fallback: tenta inferir executando um dummy crescente
        input_dim = 32

    preprocess = load_preprocess(args.preprocess)
    app = build_app(sess, input_name, input_dim, preprocess)
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()