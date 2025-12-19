"""Centralised logging configuration for the arrhythmia package.

All modules should obtain their logger via:

    from arrhythmia.utils import get_logger
    log = get_logger(__name__)
"""

from __future__ import annotations

import logging
import sys


def get_logger(name: str, level: int | str = logging.INFO) -> logging.Logger:
    """Return a logger that writes to stdout with a consistent format.

    Calling this multiple times for the same *name* is safe — the handler
    is only attached once.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def configure_root(level: int | str = logging.INFO) -> None:
    """Call once at the entry point (train.py, evaluate.py, etc.) to set
    the root logger level from config so all child loggers respect it."""
    logging.getLogger().setLevel(level)
