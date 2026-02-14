"""Logging configuration for the arrhythmia package."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str, level: int | str = logging.INFO) -> logging.Logger:
    """Return a stdout logger with a consistent format."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def configure_root(level: int | str = logging.INFO) -> None:
    """Set the root logger level."""
    logging.getLogger().setLevel(level)
