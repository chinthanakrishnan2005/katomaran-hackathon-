import logging
import os
import sys

def setup_logger():
    """
    Set up the application logger.
    Reads log file path from config if available, otherwise falls back to logs/events.log.
    Safely handles missing directories and empty dirname values.
    """
    try:
        from src.config import get_config
        config = get_config()
        log_file = config.logging.event_log
    except Exception:
        log_file = "logs/events.log"

    # Safely create log directory (guard against empty dirname)
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger("FaceTracker")
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler (INFO and above)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (INFO and above) — only if we have a valid path
    try:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception as e:
        logger.warning(f"Could not open log file '{log_file}': {e}. File logging disabled.")

    return logger


# Module-level logger instance — all modules should import this
logger = setup_logger()
