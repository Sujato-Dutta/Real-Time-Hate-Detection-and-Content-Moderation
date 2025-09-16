from __future__ import annotations

from src.data_ingestion import ingest_from_supabase
from src.data_preprocessing import preprocess_dataframe, split_dataset, DataProcessor


def main() -> None:
    # 1) Ingest data from Supabase
    df_raw = ingest_from_supabase()
    print(f"Ingested: {df_raw.shape} (rows, cols)")
    if df_raw is None or df_raw.empty:
        print("No data ingested. Exiting.")
        return

    # 2) Preprocess into standardized columns: text_clean, text_len, (optional) label
    df = preprocess_dataframe(df_raw, label_col="class")
    print(f"Preprocessed: {df.shape}")

    # 3) Split into train/val/test
    train_df, val_df, test_df = split_dataset(df, label_col="label", test_size=0.2, val_size=0.1, random_state=42)
    print(f"Splits -> train: {train_df.shape}, val: {val_df.shape}, test: {test_df.shape}")

    # 4) Fit processor on train only (leakage-safe)
    processor = DataProcessor(text_col="text_clean", label_col="label")
    processor.fit(train_df)

    # 5) Transform splits
    X_train, y_train = processor.transform(train_df)
    X_val, y_val = processor.transform(val_df)
    X_test, y_test = processor.transform(test_df)

    print(f"Vectorized -> X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    if y_train is not None:
        print(f"Labels -> y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")

    print("Data preprocessing pipeline ran successfully.")


if __name__ == "__main__":
    main()