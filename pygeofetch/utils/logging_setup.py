"""
Logging configuration for PyGeoFetch.

Sets up structured, levelled logging with optional file output and
Rich-formatted console output for CLI use.

Example::

    from pygeofetch.utils.logging_setup import setup_logging, get_logger

    setup_logging(level="DEBUG", log_file="/tmp/satellite.log")
    logger = get_logger(__name__)
    logger.info("PyGeoFetch initialized")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_LOGGING_CONFIGURED = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    use_rich: bool = True,
    quiet: bool = False,
) -> None:
    """
    Configure application-wide logging.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to write logs to file.
        use_rich: Use Rich console handler for formatted output.
        quiet: Suppress all console output (log_file still written).
    """
    global _LOGGING_CONFIGURED

    root_logger = logging.getLogger("pygeofetch")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.handlers.clear()

    if not quiet:
        if use_rich:
            try:
                from rich.logging import RichHandler
                handler = RichHandler(
                    rich_tracebacks=True,
                    show_path=level == "DEBUG",
                    markup=True,
                )
                handler.setLevel(getattr(logging, level.upper(), logging.INFO))
                root_logger.addHandler(handler)
            except ImportError:
                handler = logging.StreamHandler(sys.stderr)
                formatter = logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    datefmt="%H:%M:%S",
                )
                handler.setFormatter(formatter)
                root_logger.addHandler(handler)
        else:
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the given module name.

    Ensures the root pygeofetch logger is configured before returning.

    Args:
        name: Module name, typically __name__.

    Returns:
        Configured Logger instance.
    """
    if not _LOGGING_CONFIGURED:
        setup_logging()

    if not name.startswith("pygeofetch"):
        name = f"pygeofetch.{name}"
    return logging.getLogger(name)
