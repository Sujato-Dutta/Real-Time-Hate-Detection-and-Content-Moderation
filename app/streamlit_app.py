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
    from pathlib import Path

    def ensure_artifacts():
        """Ensure artifacts exist locally - no S3 downloads, just check local files."""
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

        required = [
            "model.joblib",
            "vectorizer.joblib", 
            "label_encoder.joblib",
            "data_processor_meta.joblib",
        ]
        
        # Check if all artifacts exist locally
        missing = [
            f for f in required
            if not os.path.exists(os.path.join(ARTIFACTS_DIR, f))
        ]
        
        if missing:
            st.error(f"❌ **Missing artifacts**: {', '.join(missing)}")
            st.error("🔧 **To fix this:**")
            st.error("1. Run locally: `python run_etl_pipeline.py`")
            st.error("2. Commit artifacts: `git add artifacts/ && git commit -m 'Add artifacts'`")
            st.error("3. Push and redeploy")
            st.stop()
        else:
            st.success(f"✅ **All {len(required)} artifacts found locally!**")

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