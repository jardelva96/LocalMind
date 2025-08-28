from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Optional, List, Union
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


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
    std = np.array(obj.get("std", []), dtype=np.float32) if "std" in obj else None
    names = obj.get("feature_names")
    labels = obj.get("class_labels")
    return {"mean": mean, "std": std, "feature_names": names, "class_labels": labels}


class PredictIn(BaseModel):
    # aceita [[...], [...]] e também lista plana [...]
    rows: Union[List[List[float]], List[float]]


class TopKItem(BaseModel):
    label: str
    prob: float


class PredictOut(BaseModel):
    preds: List[int]
    pred_labels: Optional[List[str]] = None
    probs: Optional[List[List[float]]] = None
    topk: Optional[List[List[TopKItem]]] = None


def build_app(
    session: ort.InferenceSession, input_name: str, input_dim: int, preprocess
):
    app = FastAPI(title="IAZero ONNX Server", version="0.1.0")

    # CORS (ajuste allow_origins em produção)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        # allow_credentials=True,
    )

    # Labels vindas do preprocess.json (opcional)
    class_labels: Optional[List[str]] = None
    if isinstance(preprocess, dict) and isinstance(
        preprocess.get("class_labels"), list
    ):
        class_labels = [str(x) for x in preprocess["class_labels"]]

    feature_names = (
        preprocess.get("feature_names") if isinstance(preprocess, dict) else None
    )

    @app.get("/healthz")
    def healthz():
        return {
            "status": "ok",
            "input": input_name,
            "dim": input_dim,
            "has_preprocess": preprocess is not None,
            "has_labels": class_labels is not None,
            "num_labels": len(class_labels) if class_labels else 0,
            "feature_names": feature_names,
        }

    # REGISTRE as rotas ANTES do return!
    @app.post("/predict", response_model=PredictOut)
    @app.post("/predict/", response_model=PredictOut)  # aceita com/sem barra
    def predict(inp: PredictIn, return_proba: bool = True, top_k: int = 0):
        # --- normaliza rows para 2D ---
        rows = inp.rows
        if isinstance(rows, list) and rows and isinstance(rows[0], (int, float)):
            X = np.array([rows], dtype=np.float32)  # lista plana -> 1 amostra
        else:
            X = np.array(rows, dtype=np.float32)

        # valida / ajusta dimensões
        if X.ndim != 2:
            raise ValueError(
                "Esperado matriz 2D (amostras x features) ou lista plana de floats."
            )
        if X.shape[1] > input_dim:
            X = X[:, :input_dim]
        elif X.shape[1] < input_dim:
            raise ValueError(f"Esperado {input_dim} features, recebido {X.shape[1]}.")

        # normalização opcional
        if (
            preprocess is not None
            and preprocess.get("mean") is not None
            and preprocess.get("std") is not None
        ):
            mu, sd = preprocess["mean"], preprocess["std"]
            if mu.shape[0] >= input_dim and sd.shape[0] >= input_dim:
                sd_safe = np.where(sd == 0.0, 1.0, sd)
                X = (X - mu[:input_dim]) / sd_safe[:input_dim]

        # inferência
        logits = session.run(None, {input_name: X})[0]
        preds = np.argmax(logits, axis=1).astype(int).tolist()

        # mapeia rótulos, se disponíveis
        pred_labels: Optional[List[str]] = None
        if class_labels:
            pred_labels = [
                class_labels[p] if 0 <= p < len(class_labels) else str(p) for p in preds
            ]

        probs_json: Optional[List[List[float]]] = None
        topk_json: Optional[List[List[TopKItem]]] = None

        if return_proba:
            probs = softmax(logits)
            probs_json = probs.tolist()

            if isinstance(top_k, int) and top_k > 0:
                k = max(1, min(int(top_k), probs.shape[1]))
                topk_json = []
                for i in range(probs.shape[0]):
                    idxs = np.argpartition(-probs[i], k - 1)[:k]
                    idxs = idxs[np.argsort(-probs[i, idxs])]
                    items: List[TopKItem] = []
                    for c in idxs:
                        label = (
                            class_labels[c]
                            if class_labels and 0 <= c < len(class_labels)
                            else str(int(c))
                        )
                        items.append(TopKItem(label=label, prob=float(probs[i, c])))
                    topk_json.append(items)

        return PredictOut(
            preds=preds, pred_labels=pred_labels, probs=probs_json, topk=topk_json
        )

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

    # descobre input_dim pela forma do 1º input
    shape = sess.get_inputs()[0].shape  # ex: ["batch", 32]
    input_dim = (
        int(shape[1])
        if (len(shape) >= 2 and isinstance(shape[1], (int, np.integer)))
        else 32
    )

    preprocess = load_preprocess(args.preprocess)
    app = build_app(sess, input_name, input_dim, preprocess)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
