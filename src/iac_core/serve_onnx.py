from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Optional, List, Tuple, Any
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Query
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
    def _arr(k): 
        return np.array(obj.get(k, []), dtype=np.float32) if k in obj else None
    mean = _arr("mean")
    std  = _arr("std")
    names = obj.get("feature_names")
    labels = obj.get("class_labels")  # pode ser list de ints ou strs
    return {"mean": mean, "std": std, "feature_names": names, "class_labels": labels}

class PredictIn(BaseModel):
    # aceita tanto [x1,...,xD] quanto [[...], [...]]
    rows: Any

class PredictOut(BaseModel):
    preds: List[int]
    probs: Optional[List[List[float]]] = None
    topk: Optional[List[List[Tuple[str, float]]]] = None  # [[(label, prob), ...], ...]

def normalize_rows(rows: Any) -> List[List[float]]:
    # se vier lista plana, transforma em [[...]]
    if isinstance(rows, list) and (len(rows) > 0) and all(isinstance(v, (int, float)) for v in rows):
        return [rows]
    # senão espera lista de listas
    if isinstance(rows, list) and (len(rows) > 0) and all(isinstance(v, list) for v in rows):
        return rows
    raise ValueError("Formato inválido em 'rows'. Use lista plana [..] ou lista de linhas [[..],[..]].")

def build_app(session: ort.InferenceSession, input_name: str, input_dim: int, preprocess):
    app = FastAPI(title="IAZero ONNX Server", version="0.2.0")

    labels = preprocess.get("class_labels") if preprocess else None
    if labels is not None:
        labels = [str(x) for x in labels]  # homogeniza p/ string

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "input": input_name, "dim": input_dim, "labels": labels}

    @app.post("/predict", response_model=PredictOut)
    def predict(inp: PredictIn, return_proba: bool = True, top_k: int = Query(0, ge=0)):
        rows = normalize_rows(inp.rows)
        X = np.array(rows, dtype=np.float32)

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

        out_probs = None
        out_topk = None

        if return_proba or top_k > 0:
            probs = softmax(logits)
            if return_proba:
                out_probs = probs.tolist()
            if top_k > 0:
                topk_idx = np.argpartition(-probs, kth=min(top_k, probs.shape[1]-1), axis=1)[:, :top_k]
                # ordena por prob desc
                sorted_idx = np.take_along_axis(topk_idx, np.argsort(
                    np.take_along_axis(probs, topk_idx, axis=1), axis=1)[:, ::-1], axis=1)
                out_topk = []
                for i in range(probs.shape[0]):
                    pair_list = []
                    for j in range(sorted_idx.shape[1]):
                        c = int(sorted_idx[i, j])
                        label = labels[c] if (labels is not None and c < len(labels)) else str(c)
                        pair_list.append((label, float(probs[i, c])))
                    out_topk.append(pair_list)

        return {
            "preds": preds.tolist(),
            "probs": out_probs,
            "topk": out_topk
        }

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
    shape = sess.get_inputs()[0].shape  # ex: [ 'batch', 32 ]
    input_dim = int(shape[1]) if (len(shape) >= 2 and isinstance(shape[1], (int, np.integer))) else 32

    preprocess = load_preprocess(args.preprocess)
    app = build_app(sess, input_name, input_dim, preprocess)
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
