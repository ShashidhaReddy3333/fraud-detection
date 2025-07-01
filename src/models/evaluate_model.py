"""Evaluate trained model on hold‑out and print metrics."""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve

BASE = Path(__file__).resolve().parents[2]
DATA_PATH = BASE / "data" / "processed" / "train_ready.parquet"
MODEL_PATH = BASE / "models" / "latest_model.pkl"
THRESH_PATH = BASE / "models" / "threshold.txt"

def main():
    df = pd.read_parquet(DATA_PATH)
    y = df['isFraud'].astype('int8')
    X = df.drop(columns=['isFraud'])
    split_idx = int(len(X) * 0.90)
    X_val, y_val = X.iloc[split_idx:], y.iloc[split_idx:]

    model = joblib.load(MODEL_PATH)
    thr = float(THRESH_PATH.read_text().strip())

    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs >= thr).astype(int)

    auc = roc_auc_score(y_val, probs)
    print(f"AUROC: {auc:.4f}")
    print(classification_report(y_val, preds, digits=3))
    cm = confusion_matrix(y_val, preds)
    print("Confusion matrix:\n", cm)

    fpr, tpr, _ = roc_curve(y_val, probs)
    for f, t in zip(fpr, tpr):
        if f <= 0.03:
            print(f"Recall @ {f:.3%} FPR: {t:.3%}")

if __name__ == '__main__':
    main()
