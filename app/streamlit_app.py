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
            
            # Build filename -> md5 map for required artifacts using dvc.lock entries
            file_md5_map = {}
            for fname in required:
                md5_key = f"artifacts/{fname}"
                if md5_key in md5s:
                    file_md5_map[fname] = md5s[md5_key]
            
            missing_in_lock = [fname for fname in required if fname not in file_md5_map]
            if missing_in_lock:
                raise FileNotFoundError(
                    f"Checksums for {', '.join(missing_in_lock)} not found in dvc.lock. "
                    "Run training/ETL to regenerate dvc.lock, then `dvc push`."
                )

            # Add debug logging for S3 connection
            import logging
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            
            logger.info(f"Connecting to S3: bucket={bucket}, endpoint={endpoint}")
            logger.info(f"Required artifacts: {required}")
            logger.info(f"Missing artifacts: {missing}")
            
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
            
            # First, let's list what's actually in the bucket to debug
            try:
                logger.info("Listing bucket contents to debug...")
                response = s3.list_objects_v2(Bucket=bucket, MaxKeys=50)
                if 'Contents' in response:
                    logger.info(f"Found {len(response['Contents'])} objects in bucket:")
                    for obj in response['Contents'][:10]:  # Show first 10 objects
                        logger.info(f"  - {obj['Key']} (size: {obj['Size']})")
                else:
                    logger.warning("Bucket appears to be empty or inaccessible")
            except Exception as e:
                logger.error(f"Failed to list bucket contents: {e}")

            for fname, key_md5 in file_md5_map.items():
                # DVC stores artifacts under 'artifacts/' folder in DagsHub
                # Try different possible key formats with artifacts/ prefix
                artifacts_dvc3_key = f"artifacts/files/md5/{key_md5[:2]}/{key_md5[2:]}"
                artifacts_dvc2_key = f"artifacts/md5/{key_md5[:2]}/{key_md5[2:]}"
                artifacts_root_key = f"artifacts/{key_md5}"
                
                # Also try without artifacts/ prefix (fallback)
                dvc3_key = f"files/md5/{key_md5[:2]}/{key_md5[2:]}"
                dvc2_key = f"md5/{key_md5[:2]}/{key_md5[2:]}"
                root_key = key_md5
                
                dest = os.path.join(ARTIFACTS_DIR, fname)

                logger.info(f"Downloading {fname} (md5: {key_md5})")
                success = False
                
                # Try all possible key layouts - prioritize artifacts/ prefix since that's where they are in DagsHub
                key_attempts = [
                    artifacts_dvc3_key,    # artifacts/files/md5/xx/xxxxx (DVC 3.0 with artifacts/)
                    artifacts_dvc2_key,    # artifacts/md5/xx/xxxxx (DVC 2.x with artifacts/)
                    artifacts_root_key,    # artifacts/xxxxx (root with artifacts/)
                    dvc3_key,             # files/md5/xx/xxxxx (DVC 3.0 without artifacts/)
                    dvc2_key,             # md5/xx/xxxxx (DVC 2.x without artifacts/)
                    root_key              # xxxxx (root without artifacts/)
                ]
                
                for attempt, key in enumerate(key_attempts, 1):
                    try:
                        logger.info(f"  Attempt {attempt}: trying key '{key}'")
                        # Quick existence check
                        s3.head_object(Bucket=bucket, Key=key)
                        s3.download_file(bucket, key, dest)
                        logger.info(f"  ✓ Successfully downloaded {fname} using key: {key}")
                        success = True
                        break
                    except botocore.exceptions.ClientError as e:
                        code = e.response.get("Error", {}).get("Code")
                        msg = e.response.get("Error", {}).get("Message")
                        logger.warning(f"  ✗ Key '{key}' failed: {code} - {msg}")
                        if code not in ("404", "NoSuchKey", "NotFound"):
                            # Non-404 error (auth, permissions, etc.) - don't try other keys
                            raise FileNotFoundError(
                                f"Failed to download {fname} from DagsHub S3. "
                                f"Bucket={bucket}, Endpoint={endpoint}, Key={key}. "
                                f"Error: {code} - {msg}"
                            ) from e
                    except Exception as e:
                        logger.warning(f"  ✗ Key '{key}' failed with unexpected error: {e}")
                
                if not success:
                    tried_keys = "', '".join(key_attempts)
                    raise FileNotFoundError(
                        f"Missing artifact {fname} (md5: {key_md5}) in S3. "
                        f"Tried keys: '{tried_keys}'. "
                        f"Bucket: {bucket}, Endpoint: {endpoint}. "
                        "The artifacts may not have been pushed to DVC remote. "
                        "Please run 'dvc push -r origin' locally or check if CI/CD completed successfully."
                    )
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