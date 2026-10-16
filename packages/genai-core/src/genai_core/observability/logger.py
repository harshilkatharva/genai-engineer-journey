from __future__ import annotations

import contextlib
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from .config import ObservabilityConfig
from .formatters import JsonFormatter, TextFormatter
from .redactor import RedactingFilter

# Track managed handlers to allow clean reconfiguration and shutdown
_managed_handlers: list[logging.Handler] = []


def _parse_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    lvl = getattr(logging, level.upper(), None)
    if isinstance(lvl, int):
        return lvl
    return logging.INFO


def get_logger(name: str = "genai_core") -> logging.Logger:
    """Get a namespaced logger for genai-core components.

    Attaches a NullHandler if the root genai_core logger has no handlers yet,
    preventing noisy 'No handler found' warnings before caller configuration.
    """
    logger = logging.getLogger(name)
    base_logger = logging.getLogger("genai_core")
    if not base_logger.handlers:
        base_logger.addHandler(logging.NullHandler())
    return logger


def configure_observability(config: ObservabilityConfig | None = None) -> logging.Logger:
    """Configure logging for genai-core according to caller requirements.

    Consuming applications specify the log destination, format, level, and console options.
    Re-calling this function safely closes and replaces previously registered handlers.
    """
    cfg = config or ObservabilityConfig()
    target_logger = logging.getLogger(cfg.logger_name)

    # Clean up any previously managed handlers
    close_observability(target_logger.name)

    target_logger.setLevel(_parse_level(cfg.log_level))
    target_logger.propagate = False

    # Choose formatter
    formatter: logging.Formatter
    if cfg.log_format == "json":
        formatter = JsonFormatter(redact=cfg.redact_secrets)
    else:
        formatter = TextFormatter(redact=cfg.redact_secrets)

    redacting_filter = (
        RedactingFilter(extra_patterns=cfg.extra_redaction_patterns) if cfg.redact_secrets else None
    )

    # Configure file handler if destination explicitly requested
    if cfg.log_file is not None:
        log_path = Path(cfg.log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler: logging.Handler
        if cfg.rotate_logs:
            file_handler = RotatingFileHandler(
                filename=log_path,
                maxBytes=cfg.max_bytes,
                backupCount=cfg.backup_count,
                encoding="utf-8",
            )
        else:
            file_handler = logging.FileHandler(
                filename=log_path,
                encoding="utf-8",
            )

        file_handler.setFormatter(formatter)
        if redacting_filter:
            file_handler.addFilter(redacting_filter)
        target_logger.addHandler(file_handler)
        _managed_handlers.append(file_handler)

    # Configure console handler if requested
    if cfg.console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        if redacting_filter:
            console_handler.addFilter(redacting_filter)
        target_logger.addHandler(console_handler)
        _managed_handlers.append(console_handler)

    return target_logger


def close_observability(logger_name: str = "genai_core") -> None:
    """Close and detach all managed handlers, restoring a clean state."""
    target_logger = logging.getLogger(logger_name)

    for handler in _managed_handlers:
        with contextlib.suppress(Exception):
            handler.flush()
            handler.close()
        if handler in target_logger.handlers:
            target_logger.removeHandler(handler)

    _managed_handlers.clear()

    # Retain NullHandler if no handlers exist
    if not target_logger.handlers:
        target_logger.addHandler(logging.NullHandler())


def safe_log(
    logger: logging.Logger,
    level: int,
    message: str,
    exc_info: bool | BaseException | tuple[Any, ...] | None = None,
    **extra: Any,
) -> None:
    """Safe wrapper around logger.log ensuring logging errors never break business logic."""
    with contextlib.suppress(Exception):
        logger.log(level, message, exc_info=exc_info, extra=extra)
