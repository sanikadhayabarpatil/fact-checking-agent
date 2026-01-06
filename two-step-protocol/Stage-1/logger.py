"""
logger.py

Centralized logger for the fact-checking pipeline.
"""

import logging
import sys


def _create_logger() -> logging.Logger:
    logger = logging.getLogger("fact_checker")

    if logger.handlers:
        return logger  # Prevent duplicate handlers

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = _create_logger()
