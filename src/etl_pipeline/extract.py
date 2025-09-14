import os
import kagglehub

def extract_data(dataset: str, download_path: str) -> str:
    """
    Extract dataset from Kaggle using kagglehub.

    Args:
        dataset (str): Kaggle dataset in the format 'username/dataset-name'
        download_path (str): Local path to save dataset

    Returns:
        str: Path where dataset is saved
    """

    # Download latest version from KaggleHub
    path = kagglehub.dataset_download(dataset)

    # Ensure target directory exists
    os.makedirs(download_path, exist_ok=True)

    # Move downloaded path into our raw data folder
    final_path = os.path.join(download_path, os.path.basename(path))
    if not os.path.exists(final_path):
        os.rename(path, final_path)
    return final_path
