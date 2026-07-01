"""Structured logging setup for ModelDoctor.

Provides a ``get_logger()`` factory that returns a standard :class:`logging.Logger`
with optional Rich handler for beautiful console output.

Usage::

    from modeldoctor.utils.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Starting diagnostic pipeline")
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

_INITIALIZED = False
_ROOT_LOGGER_NAME = "modeldoctor"


def setup_logging(level: str = "WARNING", rich_output: bool = True, log_file: Optional[str] = None) -> None:
    """Configure root-level logging for the modeldoctor namespace.

    Calling this multiple times is a no-op after the first successful setup.

    Args:
        level: Log level string (default WARNING to avoid console noise).
        rich_output: When ``True`` and Rich is installed, use RichHandler.
        log_file: Optional file path to output structured JSON logs.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(logging.DEBUG)  # Root catches all, handlers filter

    if not root.handlers:
        # 1. Console Handler (avoid noise)
        console_handler: logging.Handler
        if rich_output:
            try:
                from rich.logging import RichHandler
                console_handler = RichHandler(show_time=True, show_level=True, show_path=False)
                console_handler.setFormatter(logging.Formatter("%(message)s"))
            except ImportError:
                console_handler = _plain_handler()
        else:
            console_handler = _plain_handler()
            
        console_handler.setLevel(getattr(logging, level.upper(), logging.WARNING))
        root.addHandler(console_handler)
        
        # 2. Structured File Handler (detailed)
        if log_file:
            import json
            class JsonFormatter(logging.Formatter):
                def format(self, record: logging.LogRecord) -> str:
                    log_record = {
                        "timestamp": self.formatTime(record, self.datefmt),
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage()
                    }
                    if record.exc_info:
                        log_record["exc_info"] = self.formatException(record.exc_info)
                    return json.dumps(log_record)
                    
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(JsonFormatter())
            root.addHandler(file_handler)

    _INITIALIZED = True


def _plain_handler() -> logging.StreamHandler:  # type: ignore[type-arg]
    handler = logging.StreamHandler(sys.stderr)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    return handler


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger scoped to the modeldoctor namespace.

    Args:
        name: Sub-module name (typically ``__name__``).  If omitted, returns the
            root modeldoctor logger.

    Returns:
        A :class:`logging.Logger` instance.
    """
    if name and not name.startswith(_ROOT_LOGGER_NAME):
        name = f"{_ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name or _ROOT_LOGGER_NAME)
