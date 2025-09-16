# src/data_preprocessing.py
from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Any
import re
import os

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from src.utils.logger import get_logger

logger = get_logger(__name__)


# Text cleaning utilities
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_BAD_WS_RE = re.compile(r"\s+")
_HTML_ENTITY_RE = re.compile(r"&[a-zA-Z]+;|&#\d+;")


def clean_text(text: Any) -> str:
    """
    Basic, fast text cleaner suitable for social text.
    - Lowercase
    - Replace URLs with <url>
    - Replace @mentions with <user>
    - Convert #hashtag to the word itself
    - Remove common HTML entities
    - Remove extra whitespace
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text)
    s = s.lower()
    s = _URL_RE.sub(" <url> ", s)
    s = _MENTION_RE.sub(" <user> ", s)
    # Turn #word -> word
    s = _HASHTAG_RE.sub(lambda m: f" {m.group(1)} ", s)
    # Remove simple HTML entities (keep core text)
    s = _HTML_ENTITY_RE.sub(" ", s)
    # Normalize whitespace
    s = _BAD_WS_RE.sub(" ", s).strip()
    return s


# Column detection & preprocessing

def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    """Heuristically pick a likely text column (object dtype, avg length > 10)."""
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        try:
            avg_len = df[c].dropna().astype(str).str.len().mean()
            if pd.notna(avg_len) and avg_len > 10:
                return c
        except Exception:
            continue
    return obj_cols[0] if obj_cols else None


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: Optional[str] = None,
    label_col: Optional[str] = None,
    dropna_text: bool = True,
) -> pd.DataFrame:
    """
    Create a lightweight processed frame with a cleaned text column and simple metadata.

    Returns a copy with columns:
    - text_clean (str)
    - text_len (int)
    - label (if provided)
    """
    if df is None or df.empty:
        logger.warning("preprocess_dataframe: received empty DataFrame")
        return pd.DataFrame(columns=["text_clean", "text_len"] + (["label"] if label_col else []))

    if text_col is None:
        text_col = detect_text_column(df)
        if not text_col:
            raise ValueError("Could not detect a text column. Please provide text_col explicitly.")
        logger.info("Detected text column: %s", text_col)

    proc = df.copy()
    if dropna_text:
        proc = proc[proc[text_col].notna()].copy()

    proc["text_clean"] = proc[text_col].apply(clean_text)
    proc["text_len"] = proc["text_clean"].astype(str).str.len()

    if label_col and label_col in proc.columns:
        proc = proc.rename(columns={label_col: "label"})
        # Map numeric codes {0,1,2} -> {'hate_speech','offensive_language','neither'} if applicable
        uniques = set(proc["label"].astype(str).str.strip().unique().tolist())
        if uniques.issubset({"0", "1", "2"}):
            code_map = {"0": "hate_speech", "1": "offensive_language", "2": "neither"}
            proc["label"] = proc["label"].astype(str).str.strip().map(code_map)
    elif label_col:
        logger.warning("Label column '%s' not found. Proceeding without labels.", label_col)

    keep_cols = ["text_clean", "text_len"] + (["label"] if "label" in proc.columns else [])
    return proc[keep_cols]


# Dataset splitting

def split_dataset(
    df: pd.DataFrame,
    label_col: Optional[str] = "label",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train/val/test (stratified if labels available).

    Returns (train_df, val_df, test_df).
    """
    if df is None or df.empty:
        raise ValueError("split_dataset: DataFrame is empty")

    stratify = df[label_col] if (label_col and label_col in df.columns) else None

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify
    )

    stratify2 = train_df[label_col] if (label_col and label_col in train_df.columns) else None
    rel_val = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=rel_val, random_state=random_state, stratify=stratify2
    )

    logger.info(
        "Split dataset -> train: %d, val: %d, test: %d", len(train_df), len(val_df), len(test_df)
    )
    return train_df, val_df, test_df


# Vectorization & label encoding

def build_vectorizer(
    max_features: int = 50000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
) -> TfidfVectorizer:
    """Create a simple TfidfVectorizer suited for short texts."""
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        lowercase=False,  # we already lowercase in clean_text
        strip_accents="unicode",
        analyzer="word",
    )
    return vec


def vectorize_texts(vectorizer: TfidfVectorizer, texts: List[str]):
    """Fit-transform texts with the provided vectorizer (or transform if already fitted)."""
    try:
        return vectorizer.fit_transform(texts)
    except AttributeError:
        # if vectorizer is None
        raise ValueError("vectorizer must be a scikit-learn vectorizer instance")


def transform_texts(vectorizer: TfidfVectorizer, texts: List[str]):
    return vectorizer.transform(texts)


def encode_labels(labels: List[Any]) -> Tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return y, le


def decode_labels(encoded: np.ndarray, label_encoder: LabelEncoder) -> List[Any]:
    return label_encoder.inverse_transform(encoded)


def get_class_weights(y: np.ndarray, label_encoder: Optional[LabelEncoder] = None) -> Dict[Any, float]:
    """Compute balanced class weights. Returns mapping of class label to weight."""
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    mapping: Dict[Any, float] = {}
    for c, w in zip(classes, weights):
        key = label_encoder.inverse_transform([c])[0] if label_encoder is not None else int(c)
        mapping[key] = float(w)
    return mapping


# High-level processor (fit/transform/save/load)
class DataProcessor:
    """
    Lightweight processor for text datasets.
      - cleans text
      - vectorizes with TF-IDF
      - encodes labels (optional)

    Example:
        proc = DataProcessor(text_col="tweet", label_col="class")
        proc.fit(train_df)
        X_train, y_train = proc.transform(train_df)
        X_val, y_val = proc.transform(val_df)
    """

    def __init__(
        self,
        text_col: str = "text_clean",
        label_col: Optional[str] = "label",
        max_features: int = 50000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
    ) -> None:
        self.text_col = text_col
        self.label_col = label_col
        self.vectorizer = build_vectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df)
        self.label_encoder: Optional[LabelEncoder] = None
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "DataProcessor":
        if self.text_col not in df.columns:
            raise ValueError(f"DataProcessor.fit: text column '{self.text_col}' not found")
        # Guard against accidental leakage if a 'split' column is present
        if "split" in df.columns:
            uniq = set(df["split"].astype(str).str.lower().unique())
            if len(uniq) > 1:
                raise RuntimeError(
                    "DataProcessor.fit received mixed split data (found multiple 'split' values). "
                    "Pass only training rows to avoid data leakage."
                )
            # Warn if the single split value isn't 'train'
            only_val = next(iter(uniq))
            if only_val != "train":
                logger.warning(
                    "DataProcessor.fit is being called on split='%s'. "
                    "Ensure you're fitting only on training data to avoid leakage.", only_val
                )

        texts = df[self.text_col].astype(str).tolist()
        # Only fit (no need to allocate a matrix here)
        self.vectorizer.fit(texts)

        if self.label_col and self.label_col in df.columns:
            labels = df[self.label_col].astype(str).tolist()
            y, le = encode_labels(labels)
            self.label_encoder = le
            logger.info("Fitted label encoder with classes: %s", list(le.classes_))
        self.is_fitted = True
        logger.info("Vectorizer fitted on %d documents", len(texts))
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[Any, Optional[np.ndarray]]:
        if not self.is_fitted:
            raise RuntimeError("DataProcessor.transform called before fit")
        if self.text_col not in df.columns:
            raise ValueError(f"DataProcessor.transform: text column '{self.text_col}' not found")
        X = self.vectorizer.transform(df[self.text_col].astype(str).tolist())
        y = None
        if self.label_col and self.label_col in df.columns and self.label_encoder is not None:
            y = self.label_encoder.transform(df[self.label_col].astype(str).tolist())
        return X, y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[Any, Optional[np.ndarray]]:
        self.fit(df)
        return self.transform(df)

    def save(self, dir_path: str = "artifacts") -> None:
        os.makedirs(dir_path, exist_ok=True)
        dump(self.vectorizer, os.path.join(dir_path, "vectorizer.joblib"))
        if self.label_encoder is not None:
            dump(self.label_encoder, os.path.join(dir_path, "label_encoder.joblib"))
        meta = {"text_col": self.text_col, "label_col": self.label_col, "is_fitted": self.is_fitted}
        dump(meta, os.path.join(dir_path, "data_processor_meta.joblib"))
        logger.info("Saved DataProcessor artifacts to %s", dir_path)

    @classmethod
    def load(cls, dir_path: str = "artifacts") -> "DataProcessor":
        vectorizer = load(os.path.join(dir_path, "vectorizer.joblib"))
        meta = load(os.path.join(dir_path, "data_processor_meta.joblib"))
        dp = cls(text_col=meta.get("text_col", "text_clean"), label_col=meta.get("label_col"))
        dp.vectorizer = vectorizer
        label_encoder_path = os.path.join(dir_path, "label_encoder.joblib")
        if os.path.exists(label_encoder_path):
            dp.label_encoder = load(label_encoder_path)
        dp.is_fitted = bool(meta.get("is_fitted", False))
        logger.info("Loaded DataProcessor from %s", dir_path)
        return dp


__all__ = [
    "clean_text",
    "detect_text_column",
    "preprocess_dataframe",
    "split_dataset",
    "build_vectorizer",
    "vectorize_texts",
    "transform_texts",
    "encode_labels",
    "decode_labels",
    "get_class_weights",
    "DataProcessor",
]