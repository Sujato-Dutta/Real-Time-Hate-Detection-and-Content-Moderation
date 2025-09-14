# src/utils/logger.py

import logging
import os
from datetime import datetime

def get_logger(name: str) -> logging.Logger:
    """
    Sets up and returns a logger with both console and file output.
    Creates a new log file in logs/ directory for each run.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:  # Prevent duplicate handlers in interactive environments
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(module)s - %(message)s"))

        # Stream (console) handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(module)s - %(message)s"))

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
