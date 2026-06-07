import logging 
import os
import sys

# Configure logging
## define log level
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

def setup_logger(name: str, mirror_to_console: bool = False) -> logging.Logger:
    """Configures and returns a standardized logging instance."""
    logger = logging.getLogger(name)

    # If the logger has handlers, don't duplicate them
    if logger.handlers:
        return logger
    
    logger.setLevel(LOG_LEVEL)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s", 
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. CORE PERSISTENT FILE STREAM LAYER
    # Using os.path.dirname allows the code to dynamically calculate your root path regardless of where it is executed from
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(base_dir, "logs")
    
    try:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(os.path.join(log_dir, "backend.log"), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as path_err:
        # Fallback to absolute local system directory path if workspace boundaries fail
        fallback_dir = os.path.abspath("logs")
        os.makedirs(fallback_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(fallback_dir, "backend.log"), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 2. CONSOLE MIRROR LAYER
    # Fixed: Removed the duplicate unconditional StreamHandler from above to prevent stream lockups
    if mirror_to_console:
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger