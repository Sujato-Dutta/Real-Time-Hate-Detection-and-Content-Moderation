---
title: Hate Speech Detection Dashboard
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8501
pinned: false
---

# Real-Time Hate Detection and Content Moderation

A production-ready, end-to-end pipeline to detect hate speech and offensive language in short texts. It includes an ETL pipeline, Supabase data ingestion, robust preprocessing, model training and evaluation with MLflow logging,concept drift detection and handling, reproducible pipelines via DVC, a Streamlit demo app, Docker packaging, deployment in Hugging Face Spaces and automated CI/CD pipeline with GitHub Actions.

## App Link : https://huggingface.co/spaces/Sujato02/REAL_TIME_HATE_DETECTION_AND_CONTENT_MODERATION

## Features
- Supabase ingestion +  ETL
- Cleaning, TF‑IDF vectorization, label encoding
- Grid search over Logistic Regression and Random Forest
- MLflow experiment: `hate_speech_classification` (metrics, reports, confusion matrices, artifacts)
- DVC pipeline: preprocess → train → predict
- Concept drift detection and retraining
- Streamlit demo (`app/streamlit_app.py`)
- Docker image for portable deployment
- GitHub Actions CI/CD pipeline

## Getting Started (Local)
1) Setup
- Python 3.11+
- pip install -r requirements.txt

2) Configure data source
- Edit `src/config/config.yaml` with your Supabase `url`, `key`, and `table_name`.
- Do NOT commit real keys; use repository secrets in CI/CD.

3) (Optional) MLflow
- To log to a remote server, set `MLFLOW_TRACKING_URI` before running.

4) Run
- Preprocess: `python run_data_preprocessing.py`
- Train: `python run_model_train.py`
- Evaluate & drift checks: `python run_model_predict.py`
- Streamlit demo: `streamlit run app/streamlit.py`

## Reproducibility with DVC
- Full pipeline: `dvc repro`
- For the demo or Docker, ensure artifacts are present: `dvc pull` (if a remote is configured).

## CI/CD (GitHub Actions)
- Workflow at `.github/ci-cd_pipeline.yml` runs on push/PR to validate ETL/ingest, reproduce DVC stages, build/push Docker, and deploy.
- Required repository secrets (examples):
  - Supabase: `SUPABASE_URL`, `SUPABASE_KEY`
  - MLflow (DagsHub): `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`
  - Kaggle: `KAGGLE_USERNAME`, `KAGGLE_KEY`
  - Hugging Face: `HF_TOKEN`, `HF_USER`, `HF_SPACE`
  - DVC S3 (if used): `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
  - Container registry: `GITHUB_TOKEN` is provided automatically by Actions
- Trigger: commit and push to the default branch; monitor progress in the Actions tab and verify MLflow runs in DagsHub.

## Project Structure
- `src/etl_pipeline.py` — ETL pipeline for data ingestion from Kaggle and preprocessing
- `src/data_ingestion.py` — Supabase client and ingestion utilities
- `src/data_preprocessing.py` — cleaning, vectorizer, label encoder, splits, `DataProcessor`
- `src/model_train.py` — grid search, metrics & artifacts logging, saves model and preprocessing
- `src/model_predict.py` — evaluation, drift checks, MLflow logging
- `run_*.py` — entry points for each stage
- `dvc.yaml` — pipeline stages
- `app/streamlit_app.py` — demo UI
- `Dockerfile` — containerized demo with optional `dvc pull`

## Security
- `.env` and secrets must never be committed. Use repository/organization secrets in CI.
- Rotate any credentials that were exposed during development.

## Troubleshooting
- Supabase auth: verify `src/config/config.yaml` (local) or injected secrets (CI).
- MLflow logging: set `MLFLOW_TRACKING_URI` (remote) or check local `mlruns/`.
- DVC: ensure remote is configured and credentials available.
- Streamlit: confirm `artifacts/` exists (train locally or `dvc pull`).

## License
For research and educational purposes. Ensure compliance with data and platform policies when deploying.

## Author 
Sujato Dutta | [LinkedIn](https://www.linkedin.com/in/sujato-dutta/).
