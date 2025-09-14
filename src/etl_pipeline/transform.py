import pandas as pd
import os

def transform_data(raw_data_path: str, filename: str) -> pd.DataFrame:
    """
    Transform raw dataset into clean DataFrame.
    
    Args:
        raw_data_path (str): Path to raw downloaded data
        filename (str): Name of CSV file inside dataset
    
    Returns:
        pd.DataFrame: Transformed data
    """
    file_path = os.path.join(raw_data_path, filename)
    df = pd.read_csv(file_path)
    if df.columns[0].startswith("Unnamed"):
        df.rename(columns={df.columns[0]: "Id"}, inplace=True)
    
    print(f"Original shape: {df.shape}")

    # Example cleaning (adjust depending on dataset)
    df = df.dropna()
    df = df.drop_duplicates()

    print(f"Transformed shape: {df.shape}")
    return df
