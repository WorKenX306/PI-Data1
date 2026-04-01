# Project Title
PI DATA Phase 2 — 5G/6G Intrusion Detection System (IDS)

## Overview
This repository contains an academic machine learning pipeline for intrusion detection across multiple telecom/IoT datasets (5G + 6G). It includes dataset loading, preprocessing, feature selection per dataset, training for 6 models, prediction utilities, evaluation utilities, and an initial MLOps-style pipeline runner that saves trained artifacts and reports.

## Features
- Multi-dataset training (`mMTC`, `URLLC`, `eMBB`, `TON_IoT`)
- Per-dataset feature selection via `FEATURE_MAP`
- Unified preprocessing pipeline (numeric + categorical handling)
- Training for 6 model families: `rf`, `xgb`, `lr`, `et`, `mlp`, `lgbm`
- Optional imbalance handling (SMOTE when available)
- Artifact persistence (trained estimator + preprocessor + label encoder)
- Prediction utilities from in-memory results or saved artifacts
- Evaluation utilities (accuracy/precision/recall/F1 + ROC-AUC + confusion matrix + classification report)
- MLOps runner that trains all models, saves artifacts, writes metrics reports, and smoke-tests prediction from artifacts

## Tech Stack
### Frontend
- N/A (training pipeline + utilities). Optional Streamlit can be added later for UI.

### Backend
- Python
- Key libraries: numpy, pandas, scikit-learn, imbalanced-learn, xgboost, lightgbm, matplotlib, seaborn

## Architecture
Recommended minimal structure in this repo:

- `data/`
  - `Data5G/` (CSV files: `mMTC.csv`, `URLLC.csv`, `eMBB.csv`)
  - `Data6G/` (CSV file: `train_test_network.csv`)
  - `models/` (saved artifacts)
    - `<model_key>/<dataset>/artifact.pkl`
- `src/`
  - `data_loader.py` — loads datasets from `data/`
  - `features.py` — per-dataset feature lists (`FEATURE_MAP`)
  - `preprocessing.py` — `make_xy` + `build_preprocessor`
  - `config.py` — model hyperparameters (`MODEL_PARAMS`)
  - `train.py` — training + saving/loading artifacts
  - `predict.py` — prediction helpers (from TrainResult or artifact directory)
  - `evaluate.py` — evaluation metrics/reporting
  - `pipeline.py` — MLOps runner (train all models + reports)
  - Manuals:
    - `TRAIN_MANUAL.md`
    - `PREDICT_MANUAL.md`
    - `EVALUATE_MANUAL.md`
- `notebooks/`
  - `6G_IDS_final.ipynb` — original end-to-end exploration and modeling
- `reports/`
  - `mlops/` — generated metrics and checks from pipeline runs

## Contributors
- Mustapha Aziz Belkadhi
- Rami Klous
- Anas Yahyaoui
- Ahmed Karray
- zahra bendhaw
- Tasnim Kheder

## Academic Context
This project was developed as part of an academic engineering module focusing on data-driven security analytics for telecom/IoT systems (intrusion detection across 5G/6G-related datasets).

## Getting Started

### 1) Create a virtual environment (recommended)
Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Verify your data files
Expected paths:
- `data/Data5G/mMTC.csv`
- `data/Data5G/URLLC.csv`
- `data/Data5G/eMBB.csv`
- `data/Data6G/train_test_network.csv`

### 4) Train one model (and save artifacts)

```powershell
py -3 -c "from src.train import train; train('rf', dataset_name='TON_IoT')"
```

Artifacts are saved under:
- `data/models/rf/TON_IoT/artifact.pkl`

### 5) Train all models + generate reports (MLOps pipeline)

```powershell
py -3 -c "from src.pipeline import run_mlops_pipeline; run_mlops_pipeline(dataset_name='eMBB', verbose=True)"
```

Or with Make (Git Bash / WSL):

```bash
make pipeline
make train-rf DATASET=eMBB
make predict MODEL=rf DATASET=eMBB
```

On Windows, use **GNU Make from Git** if `make help` fails with `CreateProcess` / `echo` errors (avoid mixing with other Make builds). From the project folder:

`"C:\Program Files\Git\usr\bin\make.exe" help`

See `make help` for all targets.

Reports are written to:
- `reports/mlops/metrics_latest.csv`
- `reports/mlops/best_models_latest.json`
- `reports/mlops/artifact_checks_latest.csv`

### 6) Predict from saved artifacts

```python
from src.predict import predict_from_artifact_dir

out = predict_from_artifact_dir("data/models/rf/TON_IoT", X_new_df)
print(out["pred_labels"][:5])
```

### 7) Evaluate predictions

```python
from src.evaluate import evaluate_predictions

metrics = evaluate_predictions(y_true, y_pred, y_score=y_score)
print(metrics["accuracy"], metrics["f1"])
print(metrics["classification_report"])
```

## Acknowledgment
- Dataset providers and open-source libraries used in this project:
    open-source libraries:
      numpy
      pandas
      scikit-learn
      imbalanced-learn (SMOTE)
      xgboost
      lightgbm
      matplotlib
      seaborn
    Dataset providers :
      5G: ESPRIT
      6G: https://huggingface.co/datasets/codymlewis/TON_IoT_network

- Thanks too instructors/supervisors:
    Rahma Bouraoui
    Safa Cherif

