# src/utils/logger.py

import logging
import os
from datetime import datetime
import tempfile

def get_logger(name: str) -> logging.Logger:
    """
    Sets up and returns a logger with both console and file output.
    Creates a new log file in logs/ directory for each run.
    """
    # Prefer an environment-provided log dir; default to a writable temp dir
    default_tmp_logs = os.path.join(tempfile.gettempdir(), "logs")
    log_dir = os.environ.get("LOG_DIR", default_tmp_logs)

    log_file = None
    try:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    except Exception:
        # If we cannot create the directory (read-only FS), fall back to console-only
        log_file = None

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:  # Prevent duplicate handlers in interactive environments
        # Stream (console) handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(module)s - %(message)s"))
        logger.addHandler(stream_handler)

        # File handler only if we have a writable log_file
        if log_file is not None:
            try:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(module)s - %(message)s"))
                logger.addHandler(file_handler)
            except Exception:
                # Ignore file handler errors and continue with console-only
                pass

    return logger
