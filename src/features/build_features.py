"""
Feature-engineering pipeline: ordinal-encode categoricals,
impute numerics, and save artifacts.

Outputs:
    data/processed/train_ready.parquet
    models/preprocessor.pkl
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

BASE = Path(__file__).resolve().parents[2]
IN_PQ = BASE / "data" / "processed" / "train_merged.parquet"
OUT_PQ = BASE / "data" / "processed" / "train_ready.parquet"
PREPROC = BASE / "models" / "preprocessor.pkl"
PREPROC.parent.mkdir(exist_ok=True)

def main() -> None:
    df = pd.read_parquet(IN_PQ)
    y = df["isFraud"].astype("int8")
    X = df.drop(columns=["isFraud"])

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ]
    )

    print("Fitting preprocessing pipeline …")
    X_pre = preproc.fit_transform(X)

    # store as numpy → parquet via pandas
    ready_df = pd.DataFrame(
        np.column_stack([X_pre, y.values]),
        columns=[*(num_cols + cat_cols), "isFraud"],
    ).astype("float32")
    ready_df.to_parquet(OUT_PQ, index=False)
    joblib.dump(preproc, PREPROC)
    print("Saved features →", OUT_PQ)
    print("Saved preprocessor →", PREPROC)

if __name__ == "__main__":
    main()
