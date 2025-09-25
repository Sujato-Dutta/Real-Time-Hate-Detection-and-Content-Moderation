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
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

        # Map DAGSHUB_TOKEN to AWS_* if only DAGSHUB_TOKEN is provided
        if not os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("DAGSHUB_TOKEN"):
            os.environ["AWS_ACCESS_KEY_ID"] = os.environ["DAGSHUB_TOKEN"]
        if not os.environ.get("AWS_SECRET_ACCESS_KEY") and os.environ.get("DAGSHUB_TOKEN"):
            os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["DAGSHUB_TOKEN"]
        os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

        # Read DVC remote endpoint and bucket from .dvc/config
        def _read_dvc_remote():
            endpoint = None
            bucket = None
            try:
                cfg_path = Path(ROOT) / ".dvc" / "config"
                with open(cfg_path, "r") as f:
                    for raw in f:
                        line = raw.strip()
                        if line.startswith("endpointurl"):
                            endpoint = line.split("=", 1)[1].strip()
                        if line.startswith("url"):
                            url = line.split("=", 1)[1].strip()
                            if url.startswith("s3://"):
                                bucket = url.replace("s3://", "").split("/")[0]
            except Exception:
                pass
            return endpoint, bucket

        endpoint, bucket = _read_dvc_remote()
        if endpoint:
            os.environ.setdefault("AWS_ENDPOINT_URL", endpoint)
            os.environ.setdefault("AWS_ENDPOINT_URL_S3", endpoint)

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
            # Direct S3 download via boto3 using md5 checksums from dvc.lock
            try:
                import boto3
                import botocore
            except Exception as e:
                raise FileNotFoundError(
                    f"Missing artifacts: {', '.join(missing)}. "
                    "boto3 is unavailable. Install boto3 or ensure credentials in Space."
                ) from e

            if not bucket or not endpoint:
                raise FileNotFoundError(
                    f"Remote config incomplete (bucket or endpoint missing). "
                    f"Missing artifacts: {', '.join(missing)}. Check .dvc/config."
                )

            # Parse dvc.lock to build mapping: artifacts/<file> -> md5
            def _md5_map():
                m = {}
                try:
                    lock_path = Path(ROOT) / "dvc.lock"
                    lines = lock_path.read_text().splitlines()
                    current = None
                    for i, raw in enumerate(lines):
                        line = raw.strip()
                        if line.startswith("- path:"):
                            path_val = line.split(":", 1)[1].strip()
                            current = path_val  # e.g., artifacts/model.joblib
                            # Find md5 in subsequent lines
                            j = i + 1
                            while j < len(lines):
                                nxt = lines[j].strip()
                                if nxt.startswith("- path:"):
                                    break
                                if nxt.startswith("md5:"):
                                    md5 = nxt.split(":", 1)[1].strip()
                                    m[current] = md5
                                    break
                                j += 1
                except Exception:
                    pass
                return m

            md5s = _md5_map()
            s3 = boto3.client(
                "s3",
                endpoint_url=endpoint,
                region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                config=botocore.config.Config(
                    s3={"addressing_style": "path"},
                    retries={"max_attempts": 3, "mode": "standard"},
                    signature_version="s3v4",
                ),
            )

            for fname, key_md5 in file_md5_map.items():
                # Primary: DVC remote layout is 'md5/<first-two>/<rest>'
                primary_key = f"md5/{key_md5[:2]}/{key_md5[2:]}"
                # Fallback: some setups may include 'files/md5/...'
                fallback_key = f"files/md5/{key_md5[:2]}/{key_md5[2:]}"
                dest = os.path.join(ARTIFACTS_DIR, fname)

                try:
                    # quick existence check to fail fast on wrong key before download retries
                    s3.head_object(Bucket=bucket, Key=primary_key)
                    s3.download_file(bucket, primary_key, dest)
                except botocore.exceptions.ClientError as e:
                    code = e.response.get("Error", {}).get("Code")
                    msg = e.response.get("Error", {}).get("Message")
                    if code in ("404", "NoSuchKey", "NotFound"):
                        # try fallback layout once
                        try:
                            s3.head_object(Bucket=bucket, Key=fallback_key)
                            s3.download_file(bucket, fallback_key, dest)
                        except Exception as e2:
                            raise FileNotFoundError(
                                f"Missing artifact {fname} in S3. "
                                f"Tried keys: '{primary_key}' and '{fallback_key}'. "
                                f"Error: {code} - {msg}"
                            ) from e2
                    else:
                        raise FileNotFoundError(
                            f"Failed to download {fname} from DagsHub S3. "
                            f"Bucket={bucket}, Endpoint={endpoint_url}, Key={primary_key}. "
                            f"Error: {code} - {msg}"
                        ) from e
            # Re-check after S3 download
            missing = [
                f for f in required
                if not os.path.exists(os.path.join(ARTIFACTS_DIR, f))
            ]
            if missing:
                raise FileNotFoundError(
                    f"Missing artifacts after S3 download: {', '.join(missing)}. "
                    "Verify artifacts exist in the DVC remote (dvc push)."
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