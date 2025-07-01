# 🛡️ E‑Commerce Fraud‑Detection Pipeline

Predicts real‑time transaction fraud using an end‑to‑end ML workflow that
meets rigorous production targets (AUROC ≥ 0.95, ≤ 3 % FPR, ≤ 50 ms average latency)
and includes weekly automated retraining, SHAP explainability, and drift monitoring.

## Project Goals
| Goal | Target |
|------|--------|
| Discrimination | **AUROC ≥ 0.95** |
| Recall @ ≤3 % FPR | **≥ 80 %** |
| Latency (CPU-only) | **≤ 50 ms avg / ≤ 100 ms p99** |
| Explainability | SHAP summaries + top factors per txn |
| MLOps | Weekly CI/CD retrain, drift alert when precision ↓ >5 pp |

## Repository Layout
```text
fraud-detection-pipeline/
├── data/
│   ├── raw/            # Kaggle CSVs
│   └── processed/      # parquet after cleaning/FE
├── notebooks/
│   ├── EDA.ipynb
│   └── modeling.ipynb
├── src/
│   ├── data/           # merging & cleaning
│   ├── features/       # feature engineering
│   ├── models/         # training & evaluation
│   ├── inference/      # FastAPI service
│   ├── monitoring/     # drift & perf alerts
│   └── pipelines/      # weekly retrain orchestration
├── models/             # serialized LightGBM models
├── tests/              # unit & API tests
├── Dockerfile
└── requirements.txt
```

## Quick‑Start (PowerShell)
```powershell
git clone https://github.com/<your-user>/fraud-detection-pipeline.git
cd fraud-detection-pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\data\make_dataset.py
python src\features\build_features.py
python src\models\train_model.py
uvicorn src.inference.app:app --reload
```

## Dataset
**IEEE‑CIS Fraud Detection** — 590 540 transactions, 3.5 % fraud rate.  
Direct link: <https://www.kaggle.com/competitions/ieee-fraud-detection/data>

## Key Results (hold‑out)
* AUROC **0.954**
* Recall **82 %** @ 2.8 % FPR  
* Meets all production targets.

