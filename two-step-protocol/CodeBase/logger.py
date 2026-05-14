"""
logger.py

Centralized logger for the fact-checking pipeline.
Exports:
- logger (singleton)  -> for `from logger import logger`
- get_logger(name)    -> for `from logger import get_logger`
"""

import logging
import sys
from typing import Dict

_LOGGER_CACHE: Dict[str, logging.Logger] = {}


def _create_logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    if lg.handlers:
        return lg  # Prevent duplicate handlers

    lg.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
    )
    handler.setFormatter(formatter)

    lg.addHandler(handler)
    lg.propagate = False
    return lg


def get_logger(name: str) -> logging.Logger:
    if name not in _LOGGER_CACHE:
        _LOGGER_CACHE[name] = _create_logger(name)
    return _LOGGER_CACHE[name]


# Backward-compatible singleton
logger = get_logger("fact_checker")

__all__ = ["logger", "get_logger"]
