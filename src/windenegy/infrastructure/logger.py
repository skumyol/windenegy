"""Structured logging configuration.

Provides a unified logging interface using structlog for structured output
and standard library logging compatibility.
"""

from __future__ import annotations

import logging
import sys
from typing import cast

import structlog

from windenegy.infrastructure.config import LoggingConfig


def configure_logging(config: LoggingConfig | None = None) -> None:
    """Configure structured logging for the application.

    This should be called once at application startup.

    Args:
        config: Logging configuration. If None, uses defaults.
    """
    if config is None:
        config = LoggingConfig()

    # Configure standard library logging
    level = getattr(logging, config.level)

    if config.format == "json":
        _configure_json_logging(level)
    else:
        _configure_console_logging(level)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            (
                structlog.processors.format_exc_info
                if config.include_traceback
                else structlog.processors.UnicodeDecoder()
            ),
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def _configure_json_logging(level: int) -> None:
    """Configure JSON-formatted logging."""
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


def _configure_console_logging(level: int) -> None:
    """Configure human-readable console logging."""
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=True),
        foreign_pre_chain=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M.%S"),
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name, typically __name__.

    Returns:
        Configured BoundLogger instance.
    """
    return cast("structlog.stdlib.BoundLogger", structlog.get_logger(name))
