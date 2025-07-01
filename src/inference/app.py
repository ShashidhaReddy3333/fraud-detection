"""FastAPI real‑time inference service for fraud detection."""
from fastapi import FastAPI
import joblib, pandas as pd, os, time
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE / "models" / "latest_model.pkl"))
PREPROC_PATH = os.getenv("PREPROC_PATH", str(BASE / "models" / "preprocessor.pkl"))
THRESH_PATH = os.getenv("THRESH_PATH", str(BASE / "models" / "threshold.txt"))
THRESHOLD = float(Path(THRESH_PATH).read_text().strip()) if Path(THRESH_PATH).exists() else 0.5

app = FastAPI(title="Fraud Detection API", version="1.0")

@app.on_event("startup")
def _load_artifacts():
    global model, preproc
    model = joblib.load(MODEL_PATH)
    preproc = joblib.load(PREPROC_PATH)

def preprocess(txn_json):
    df = pd.DataFrame([txn_json])
    return preproc.transform(df)

@app.post("/predict")
async def predict(txn: dict):
    start = time.perf_counter()
    X = preprocess(txn)
    proba = model.predict_proba(X)[0, 1]
    flag = int(proba >= THRESHOLD)
    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "fraud_probability": proba,
        "fraud_flag": flag,
        "latency_ms": latency_ms
    }
