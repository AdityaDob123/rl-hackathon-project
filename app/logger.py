"""Centralized structured logging for TradeDesk OpenEnv."""
from __future__ import annotations

import logging
import sys
from typing import Optional

from app import config


_FORMAT = "[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def _resolve_level() -> int:
    return _LEVEL_MAP.get(config.LOG_LEVEL.lower(), logging.INFO)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger configured from the project's LOG_LEVEL setting."""
    logger = logging.getLogger(name or "tradedesk")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FMT))
        logger.addHandler(handler)
        logger.setLevel(_resolve_level())
        logger.propagate = False
    return logger
