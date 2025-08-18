from loguru import logger
import sys
from config import LOG_LEVEL

def setup_logger():
    logger.remove()  # Clear default handlers
    logger.add(sys.stderr, level="DEBUG", format="{time} {level} {message}")
    return logger