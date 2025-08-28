import os, json, requests

BASE = os.getenv("IAZERO_URL", "http://127.0.0.1:8000")


def test_healthz():
    r = requests.get(f"{BASE}/healthz")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert d["dim"] >= 1


def test_predict_topk():
    rows = [[(i % 10) / 10.0 for i in range(32)]]
    r = requests.post(
        f"{BASE}/predict",
        params={"return_proba": True, "top_k": 3},
        headers={"Content-Type": "application/json"},
        data=json.dumps({"rows": rows}),
    )
    assert r.status_code == 200
    d = r.json()
    assert "preds" in d and len(d["preds"]) == 1
    assert "topk" in d and len(d["topk"][0]) == 3
