import sys
import time
from src.etl_pipeline.extract import extract_data
from src.etl_pipeline.transform import transform_data
from src.etl_pipeline.load import load_data
from src.utils.logger import get_logger
from src.utils.exceptions import DataIngestionError, DataTransformationError, ProjectBaseError

logger = get_logger(__name__)

def run_pipeline():
    # CONFIGS
    dataset = "yashdogra/toxic-tweets"   # replace with Kaggle dataset
    raw_data_path = "data/raw"
    file_name = "labeled_data.csv"      # update based on Kaggle dataset
    table_name = "Hate Speech Table"

    try:
        logger.info("Starting ETL pipeline...")
        start_time = time.time()

        # Extract
        step_start = time.time()
        logger.info("Step 1: Extracting data from Kaggle dataset: %s", dataset)
        try:
            path = extract_data(dataset, raw_data_path)
        except Exception as e:
            raise DataIngestionError(f"Extraction failed: {e}")
        logger.info("Data extracted to %s (%.2fs)", path, time.time() - step_start)

        # Transform
        step_start = time.time()
        logger.info("Step 2: Transforming dataset: %s", file_name)
        try:
            df = transform_data(path, file_name)
        except Exception as e:
            raise DataTransformationError(f"Transformation failed: {e}")
        logger.info("Transformation complete. DataFrame shape: %s (%.2fs)", df.shape, time.time() - step_start)

        # Load
        step_start = time.time()
        logger.info("Step 3: Loading data into Supabase table: %s", table_name)
        try:
            load_data(df, table_name)
        except Exception as e:
            raise ProjectBaseError(f"Loading failed: {e}")
        logger.info("✅ Data successfully loaded into Supabase table: %s (%.2fs)", table_name, time.time() - step_start)

        logger.info("ETL pipeline completed successfully in %.2fs!", time.time() - start_time)

    except ProjectBaseError as e:
        logger.error("Pipeline failed with project error: %s", str(e), exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error("Pipeline failed with unexpected error: %s", str(e), exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()