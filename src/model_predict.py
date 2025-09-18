from __future__ import annotations

import os
import math
import tempfile
from typing import Dict, Any, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from dotenv import load_dotenv

import mlflow
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
    """Configure MLflow using environment variables from .env (same experiment as training)."""
    load_dotenv()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("hate_speech_classification")


def _log_classification_report(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], artifact_name: str) -> Dict[str, Any]:
    """Create and log a text classification report to MLflow; return macro metrics."""
    report_text = classification_report(y_true, y_pred, target_names=labels, digits=4)
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
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
    }


def _log_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], artifact_name: str) -> None:
    """Log a confusion matrix plot to MLflow as an artifact."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    with tempfile.TemporaryDirectory() as tmpdir:
        cm_path = os.path.join(tmpdir, f"{artifact_name}.png")
        fig.savefig(cm_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(cm_path, artifact_path="plots")


def _psi(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between ref and cur numeric distributions (quantile bins on ref)."""
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)

    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if ref.size == 0 or cur.size == 0:
        return float("nan")

    # quantile-based bin edges from ref
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(ref, quantiles))
    if edges.size < 3:  # fallback to linear bins if degenerate
        edges = np.linspace(ref.min(), ref.max(), n_bins + 1)

    eps = 1e-8
    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)

    ref_prop = ref_hist / max(ref_hist.sum(), 1)
    cur_prop = cur_hist / max(cur_hist.sum(), 1)

    psi = 0.0
    for p_ref, p_cur in zip(ref_prop, cur_prop):
        p_ref = max(p_ref, eps)
        p_cur = max(p_cur, eps)
        psi += (p_cur - p_ref) * math.log(p_cur / p_ref)
    return float(psi)


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), eps)
    q = q / max(q.sum(), eps)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence between two discrete distributions p and q (0 => identical)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), eps)
    q = q / max(q.sum(), eps)
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m, eps=eps) + 0.5 * _kl_divergence(q, m, eps=eps)


def _load_artifacts(artifacts_dir: str) -> Tuple[DataProcessor, Any]:
    processor = DataProcessor.load(artifacts_dir)
    model_path = os.path.join(artifacts_dir, "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Saved model not found at {model_path}. Train first.")
    model = load(model_path)
    return processor, model


def main() -> None:
    """
    Evaluate the saved model on the test split from the current dataset, log metrics to MLflow,
    compute drift metrics (PSI on text_len and JS on label distribution), and optionally
    trigger automatic retraining when thresholds are violated.
    """
    _setup_mlflow()

    # thresholds and behavior via env
    load_dotenv()
    PSI_THRESHOLD = float(os.getenv("DRIFT_PSI_THRESHOLD", "0.2"))
    JS_THRESHOLD = float(os.getenv("DRIFT_JS_THRESHOLD", "0.1"))
    F1_THRESHOLD = float(os.getenv("RETRAIN_F1_THRESHOLD", "0.65"))
    ENABLE_AUTOMATIC_RETRAIN = os.getenv("ENABLE_AUTOMATIC_RETRAIN", "1") == "1"

    artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))
    try:
        processor, model = _load_artifacts(artifacts_dir)
    except Exception as e:
        logger.error("Failed to load artifacts: %s", str(e), exc_info=True)
        return

    logger.info("Ingesting latest data and preparing test split for evaluation...")
    df_raw = ingest_from_supabase()
    if df_raw is None or df_raw.empty:
        logger.error("No data ingested; aborting evaluation.")
        return

    # Preprocess incoming data (maps numeric class to semantic labels internally)
    df = preprocess_dataframe(df_raw, label_col="class")

    # Recreate split with same random_state used during training
    train_df, val_df, test_df = split_dataset(df, label_col="label", test_size=0.2, val_size=0.1, random_state=42)

    labels = list(processor.label_encoder.classes_) if processor.label_encoder is not None else None
    if labels is None:
        logger.error("Processor label encoder missing. Re-train first.")
        return

    # Evaluate on test
    X_test, y_test = processor.transform(test_df)
    if y_test is None:
        logger.error("No labels found in test split; cannot evaluate.")
        return

    with mlflow.start_run(run_name="evaluate_and_drift_check"):
        mlflow.log_param("artifacts_dir", artifacts_dir)

        y_pred = model.predict(X_test)
        test_metrics = _log_classification_report(y_test, y_pred, labels, artifact_name="test_classification_report")
        _log_confusion_matrix(y_test, y_pred, labels, artifact_name="test_confusion_matrix")
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in test_metrics.items()})
        logger.info("Test metrics: %s", test_metrics)

        # Drift metrics (train vs test)
        psi_text_len = _psi(train_df["text_len"].to_numpy(), test_df["text_len"].to_numpy(), n_bins=10)
        mlflow.log_metric("drift_psi_text_len_train_vs_test", float(psi_text_len))

        train_counts = train_df["label"].value_counts().reindex(labels).fillna(0).to_numpy()
        test_counts = test_df["label"].value_counts().reindex(labels).fillna(0).to_numpy()
        js_label = _js_divergence(train_counts, test_counts)
        mlflow.log_metric("drift_js_label_train_vs_test", float(js_label))

        drift_flags = {
            "psi_text_len_exceeds": bool(psi_text_len >= PSI_THRESHOLD if not np.isnan(psi_text_len) else False),
            "js_label_exceeds": bool(js_label >= JS_THRESHOLD),
            "f1_below_threshold": bool(test_metrics["f1_macro"] < F1_THRESHOLD),
        }
        mlflow.log_params({
            "psi_threshold": PSI_THRESHOLD,
            "js_threshold": JS_THRESHOLD,
            "f1_threshold": F1_THRESHOLD,
        })
        mlflow.set_tags({f"drift_flag__{k}": str(v).lower() for k, v in drift_flags.items()})

        # Performance-only retrain policy
        retrain_needed = drift_flags["f1_below_threshold"]
        mlflow.set_tag("retrain_needed", str(retrain_needed).lower())
        logger.info(
            "Flags: %s | performance-only retrain policy => retrain_needed=%s",
            drift_flags,
            retrain_needed,
        )

    # Optional: trigger retrain
    if retrain_needed and ENABLE_AUTOMATIC_RETRAIN:
        logger.warning("Triggering retrain because drift/performance thresholds violated...")
        try:
            from src.model_train import main as train_main
            train_main()
        except Exception as e:
            logger.error("Automatic retrain failed: %s", str(e), exc_info=True)
    else:
        if retrain_needed:
            logger.warning(
                "Retrain needed but ENABLE_AUTOMATIC_RETRAIN=0. Set it to 1 to auto-trigger training."
            )


if __name__ == "__main__":
    main()