from __future__ import annotations

import os
from typing import List, Optional

import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Ensure project root is on sys.path so `from src...` works when launched from any directory
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_preprocessing import DataProcessor, preprocess_dataframe

ARTIFACTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")

@st.cache_resource(show_spinner=False)
def load_artifacts():
    import subprocess
    import os

    def ensure_artifacts():
        # Map DAGSHUB_TOKEN to AWS_* if Space provided only DAGSHUB_TOKEN
        if not os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("DAGSHUB_TOKEN"):
            os.environ["AWS_ACCESS_KEY_ID"] = os.environ["DAGSHUB_TOKEN"]
        if not os.environ.get("AWS_SECRET_ACCESS_KEY") and os.environ.get("DAGSHUB_TOKEN"):
            os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["DAGSHUB_TOKEN"]
        # Optional default region for S3 clients (safe default)
        os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

        required = [
            "model.joblib",
            "vectorizer.joblib",
            "label_encoder.joblib",
            "data_processor_meta.joblib",
        ]
        missing = [
            f for f in required
            if not os.path.exists(os.path.join(ARTIFACTS_DIR, f))
        ]
        if missing:
            try:
                # Try to pull once; relies on Space env vars
                subprocess.run(["dvc", "pull", "-v"], check=True)
            except subprocess.CalledProcessError as e:
                # Surface a clear message; logs will show the error
                raise FileNotFoundError(
                    f"DVC pull failed. Missing artifacts: {', '.join(missing)}. "
                    "Ensure Space Secrets include AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
                    "(you can set both to your DagsHub token)."
                ) from e

            # Re-check after pull
            missing = [
                f for f in required
                if not os.path.exists(os.path.join(ARTIFACTS_DIR, f))
            ]
            if missing:
                raise FileNotFoundError(
                    f"Missing artifacts after DVC pull: {', '.join(missing)}. "
                    "Verify the artifacts were pushed to the DVC remote and that Space Secrets are correct."
                )

    ensure_artifacts()

    processor = DataProcessor.load(ARTIFACTS_DIR)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model artifact not found. Please train first.")
    model = load(MODEL_PATH)
    return processor, model


def predict_texts(texts: List[str]):
    processor, model = load_artifacts()
    df = pd.DataFrame({"text": texts})
    df_clean = preprocess_dataframe(df, label_col=None)
    X, _ = processor.transform(df_clean)
    labels = model.predict(X)
    probs = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)
        except Exception:
            probs = None

    # Map labels back to strings if encoder exists
    if processor.label_encoder is not None:
        try:
            labels = processor.label_encoder.inverse_transform(labels)
        except Exception:
            pass
    labels = [str(x) for x in labels]
    return labels, probs


# UI
st.set_page_config(page_title="Hate Speech Detection Dashboard", page_icon="🛡️", layout="wide")

st.title("🛡️ Hate Speech Detection Dashboard")
st.write("Classify text as hate/offensive/neutral and explore predictions.")

with st.sidebar:
    st.header("About")
    st.markdown("""
    This app uses a trained NLP model to detect hate speech. 
    - Artifacts are loaded from `artifacts/`
    - Predictions are generated on-the-fly
    """)

# Tabs
tab_pred, tab_info = st.tabs(["Text Category Prediction", "Info"])

with tab_pred:
    st.subheader("Text Prediction")
    text = st.text_area("Enter text to classify", height=150, placeholder="Type or paste text here...")
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Predict", type="primary"):
            if not text.strip():
                st.warning("Please enter some text.")
            else:
                with st.spinner("Predicting..."):
                    labels, probs = predict_texts([text.strip()])
                st.success(f"Prediction: {labels[0]}")
                if probs is not None:
                    st.caption("Class probabilities")
                    proba_df = pd.DataFrame(probs, columns=getattr(load_artifacts()[0].label_encoder, 'classes_', None))
                    st.bar_chart(proba_df.T)

with tab_info:
    st.subheader("App info")
    st.info(
        "Use the Prediction tab to classify a piece of text.\n"
        "Artifacts are loaded from the artifacts/ directory.\n"
    )

st.markdown("---")
st.caption("Built with Streamlit")