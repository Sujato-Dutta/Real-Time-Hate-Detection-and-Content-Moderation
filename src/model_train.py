from __future__ import annotations

import os
import tempfile
from typing import Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
from dotenv import load_dotenv

import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    ConfusionMatrixDisplay,
)

from src.data_ingestion import ingest_from_supabase
from src.data_preprocessing import preprocess_dataframe, split_dataset, DataProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _setup_mlflow() -> None:
    """Configure MLflow using environment variables from .env."""
    load_dotenv()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    # Set a consistent experiment name
    mlflow.set_experiment("hate_speech_classification")


def _log_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], artifact_name: str) -> None:
    """Log a confusion matrix plot to MLflow as an artifact."""
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    with tempfile.TemporaryDirectory() as tmpdir:
        cm_path = os.path.join(tmpdir, f"{artifact_name}.png")
        fig.savefig(cm_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(cm_path, artifact_path="plots")


def _log_classification_report(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], artifact_name: str) -> Dict[str, Any]:
    """Create and log a text classification report to MLflow; return macro metrics."""
    report_text = classification_report(y_true, y_pred, target_names=labels, digits=4)
    # Extract key aggregate metrics as numbers as well
    f1_macro = f1_score(y_true, y_pred, average="macro")
    precision_macro = precision_score(y_true, y_pred, average="macro")
    recall_macro = recall_score(y_true, y_pred, average="macro")
    accuracy = accuracy_score(y_true, y_pred)

    with tempfile.TemporaryDirectory() as tmpdir:
        rp_path = os.path.join(tmpdir, f"{artifact_name}.txt")
        with open(rp_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        mlflow.log_artifact(rp_path, artifact_path="reports")

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
    }


def _build_search_spaces() -> Dict[str, Tuple[Pipeline, Dict[str, list[Any]]]]:
    """Define model pipelines and their grids."""
    # Logistic Regression directly on sparse TF-IDF
    lr_pipe = Pipeline([
        ("clf", LogisticRegression(solver="saga", multi_class="multinomial", max_iter=2000, n_jobs=-1, class_weight="balanced", random_state=42)),
    ])
    lr_grid = {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__penalty": ["l2"],
    }

    # RandomForest on reduced dense representation via TruncatedSVD
    rf_pipe = Pipeline([
        ("svd", TruncatedSVD(n_components=300, random_state=42)),
        ("clf", RandomForestClassifier(random_state=42, n_jobs=-1)),
    ])
    rf_grid = {
        "svd__n_components": [100],
        "clf__n_estimators": [50, 100],
        "clf__max_depth": [None, 20],
        "clf__max_features": ["sqrt"],
    }

    return {
        "logistic_regression": (lr_pipe, lr_grid),
        "random_forest": (rf_pipe, rf_grid),
    }


def main() -> None:
    _setup_mlflow()

    logger.info("Starting training: ingest -> preprocess -> split")
    df_raw = ingest_from_supabase()
    if df_raw is None or df_raw.empty:
        logger.error("No data ingested; aborting training.")
        return

    df = preprocess_dataframe(df_raw, label_col="class")  # maps 0/1/2 to semantic labels
    train_df, val_df, test_df = split_dataset(df, label_col="label", test_size=0.2, val_size=0.1, random_state=42)

    # Fit processor on train only
    processor = DataProcessor(text_col="text_clean", label_col="label")
    processor.fit(train_df)

    # Transform splits
    X_train, y_train = processor.transform(train_df)
    X_val, y_val = processor.transform(val_df)
    X_test, y_test = processor.transform(test_df)

    labels = list(processor.label_encoder.classes_) if processor.label_encoder is not None else None
    if labels is None:
        logger.error("Labels are missing; ensure label column is present after preprocessing.")
        return

    # Hyperparameter tuning on train set; evaluate on val for model selection
    search_spaces = _build_search_spaces()

    with mlflow.start_run(run_name="rf_vs_lr_gridsearch"):
        best_name = None
        best_model = None
        best_val_f1 = -np.inf
        best_val_metrics: Dict[str, float] = {}

        for name, (pipe, grid) in search_spaces.items():
            logger.info("GridSearchCV for %s", name)
            gs = GridSearchCV(
                estimator=pipe,
                param_grid=grid,
                scoring="f1_macro",
                cv=2,
                n_jobs=-1,
                verbose=1,
            )
            gs.fit(X_train, y_train)
            logger.info("%s best params: %s", name, gs.best_params_)

            # Evaluate on validation set
            y_val_pred = gs.best_estimator_.predict(X_val)
            val_metrics = _log_classification_report(y_val, y_val_pred, labels, artifact_name=f"{name}_val_classification_report")
            _log_confusion_matrix(y_val, y_val_pred, labels, artifact_name=f"{name}_val_confusion_matrix")

            # Log params and val metrics for this candidate
            mlflow.log_params({f"{name}__{k}": v for k, v in gs.best_params_.items()})
            mlflow.log_metrics({f"val_{name}_{k}": float(v) for k, v in val_metrics.items()})

            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_macro"]
                best_name = name
                best_model = gs.best_estimator_
                best_val_metrics = val_metrics

        if best_model is None:
            logger.error("No model was selected; aborting.")
            return

        logger.info("Best model on validation: %s (f1_macro=%.4f)", best_name, best_val_f1)
        mlflow.log_param("best_model_name", best_name)
        mlflow.log_metrics({f"val_best_{k}": float(v) for k, v in best_val_metrics.items()})

        # Retrain best model on train+val, evaluate on test
        from pandas import concat as pd_concat
        trainval_df = pd_concat([train_df, val_df], ignore_index=True)
        X_trainval, y_trainval = processor.transform(trainval_df)
        best_model.fit(X_trainval, y_trainval)

        y_test_pred = best_model.predict(X_test)
        test_metrics = _log_classification_report(y_test, y_test_pred, labels, artifact_name="best_test_classification_report")
        _log_confusion_matrix(y_test, y_test_pred, labels, artifact_name="best_test_confusion_matrix")
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in test_metrics.items()})

        # Persist artifacts locally
        artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))
        os.makedirs(artifacts_dir, exist_ok=True)

        # Save trained model
        model_path = os.path.join(artifacts_dir, "model.joblib")
        dump(best_model, model_path)
        logger.info("Saved best model to %s", model_path)
        mlflow.log_artifact(model_path, artifact_path="models")

        # Save processor artifacts (vectorizer, label encoder, meta)
        from pathlib import Path
        # Save to artifacts_dir/data_processor_*
        processor.save(artifacts_dir)
        # Log saved processor files to MLflow
        for fname in ["vectorizer.joblib", "label_encoder.joblib", "data_processor_meta.joblib"]:
            fpath = os.path.join(artifacts_dir, fname)
            if os.path.exists(fpath):
                mlflow.log_artifact(fpath, artifact_path="preprocessing")

        logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()