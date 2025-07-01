"""Merge IEEE‑CIS transaction & identity CSVs → single parquet.

Usage:
    python -m src.data.make_dataset
"""
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    tx_path = RAW_DIR / "train_transaction.csv"
    id_path = RAW_DIR / "train_identity.csv"
    if not tx_path.exists():
        raise FileNotFoundError(f"Raw transactions not found: {tx_path}")
    print("Loading raw CSVs ...")
    tx = pd.read_csv(tx_path)
    id_ = pd.read_csv(id_path)
    print(f"transactions: {tx.shape}, identity: {id_.shape}")

    df = tx.merge(id_, how='left', on='TransactionID')
    print("Merged:", df.shape)

    out_path = OUT_DIR / "train_merged.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved merged parquet → {out_path}")

if __name__ == '__main__':
    main()
