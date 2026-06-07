import logging 
import os
import sys

# Configure logging
## define log level
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

def setup_logger(name: str) -> logging.Logger:
    """congifures and returns a standerdized logging instance."""
    logger = logging.getLogger(name)

    # if the logger has handeler, don't duplicate them
    if logger.handlers:
        return logger
    
    logger.setLevel(LOG_LEVEL)

    formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # streamHandler prints directly to your terminal console(visible in the docker-compose logs)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    #Filehandler saves logs to the disk inside the container
    log_dir = "/logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

