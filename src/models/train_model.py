"""
Train a LightGBM fraud-detection model with Optuna hyper-parameter search.

Outputs
-------
models/latest_model.pkl   – final fitted model
models/threshold.txt      – prob. cutoff giving 3 % FPR on the val slice
"""

from pathlib import Path

import joblib
import numpy as np
import optuna
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


# ───────────────────────────── paths ──────────────────────────────
BASE        = Path(__file__).resolve().parents[2]
DATA_PATH   = BASE / "data" / "processed" / "train_ready.parquet"
MODEL_PATH  = BASE / "models" / "latest_model.pkl"
THRESH_PATH = BASE / "models" / "threshold.txt"

# ─────────────────────── helper functions ─────────────────────────
def load_data():
    """Read processed parquet → X, y."""
    df = pd.read_parquet(DATA_PATH)
    y = df.pop("isFraud").astype("int8")
    return df, y


def time_split(X, y, val_frac: float = 0.10):
    """Hold out the most-recent `val_frac` rows (data is time-ordered)."""
    split_idx = int(len(X) * (1.0 - val_frac))
    return (
        X.iloc[:split_idx],
        X.iloc[split_idx:],
        y.iloc[:split_idx],
        y.iloc[split_idx:],
    )


# ───────────────────────── optuna objective ───────────────────────
def objective(trial, X_tr, y_tr, X_val, y_val, pos_weight: float):
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255, step=32),
        "max_depth": trial.suggest_int("max_depth", 6, 16),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
        "scale_pos_weight": pos_weight,
        # n_estimators fixed; early-stopping will pick best_iter_
        "n_estimators": 1000,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)


# ────────────────────────────── main ──────────────────────────────
def main():
    # 1. load + split
    X, y = load_data()
    X_tr, X_val, y_tr, y_val = time_split(X, y)
    pos_weight = (len(y_tr) - y_tr.sum()) / y_tr.sum()
    print("Train/Val shapes:", X_tr.shape, X_val.shape)

    # 2. Optuna search
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: objective(t, X_tr, y_tr, X_val, y_val, pos_weight),
        n_trials=30,
        show_progress_bar=True,
    )
    best = study.best_params
    best.update(
        dict(
            objective="binary",
            metric="auc",
            verbosity=-1,
            boosting_type="gbdt",
            scale_pos_weight=pos_weight,
            n_estimators=1000,
        )
    )
    print("Best params →", best)

    # 3. retrain on full train (w/ early-stopping on val)
    model = lgb.LGBMClassifier(**best)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    joblib.dump(model, MODEL_PATH)
    print("Saved model →", MODEL_PATH)

    # 4. choose threshold for 3 % FPR
    val_probs = model.predict_proba(X_val)[:, 1]
    fpr, tpr, thr = roc_curve(y_val, val_probs)
    cutoff = thr[np.where(fpr <= 0.03)[0][-1]]  # last thr under 3 %
    THRESH_PATH.write_text(f"{cutoff:.6f}")
    print(f"Saved threshold {cutoff:.4f} →", THRESH_PATH)

    # 5. final metrics
    auc = roc_auc_score(y_val, val_probs)
    recall_at_cut = tpr[np.where(fpr <= 0.03)[0][-1]]
    print(f"Validation AUROC = {auc:.4f}")
    print(f"Recall @ 3 % FPR = {recall_at_cut:.3%}")


if __name__ == "__main__":
    main()
