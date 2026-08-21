from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_app.observability.context import get_request_id


# ---------------------------------------------------------
# Log file location
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "rag_app.log"


# ---------------------------------------------------------
# JSON Formatter
# ---------------------------------------------------------


class JsonFormatter(logging.Formatter):
    """Convert logging records into controlled JSON."""

    # Only these fields are allowed from our application.
    STRUCTURED_FIELDS = (
        # Request
        "event",
        "span_id",
        "component",
        "endpoint",
        "tenant_id",
        "status",
        # Timing
        "latency_ms",
        # Query
        "technique",
        # Embedding
        "provider",
        "model",
        "token_count",
        # Retrieval
        "top_k",
        "no_of_chunks",
        # Prompt
        "prompt_version",
        "prompt_name",
        # LLM
        "input_tokens",
        "output_tokens",
        # Cost
        "cost_usd",
        # Errors
        "error_type",
        "error_message",
    )

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now(timezone.utc).isoformat()
        request_id = getattr(record, "request_id", None)

        log_data: dict[str, Any] = {
            "timestamp": timestamp,
            "request_id": request_id,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Only copy fields that OUR application explicitly supports.
        for field in self.STRUCTURED_FIELDS:
            value = getattr(record, field, None)

            if value is not None:
                log_data[field] = value

        # Add exception details when logger.exception() is used.
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(
            log_data,
            default=str,
            ensure_ascii=False,
        )


# ---------------------------------------------------------
# Logger
# ---------------------------------------------------------


class ObservabilityLogger:
    """Central application logger."""

    def __init__(
        self,
        name: str = "rag_app",
        level: int = logging.INFO,
    ) -> None:
        self._logger = logging.getLogger(name)

        self._logger.setLevel(level)

        # Don't pass logs to the root logger.
        self._logger.propagate = False

        # Prevent duplicate handlers.
        if self._logger.handlers:
            return

        formatter = JsonFormatter()

        # -------------------------------------------------
        # Terminal
        # -------------------------------------------------

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        # -------------------------------------------------
        # File
        # -------------------------------------------------

        file_handler = logging.FileHandler(
            LOG_FILE,
            encoding="utf-8",
        )

        file_handler.setFormatter(formatter)

        # -------------------------------------------------
        # Register handlers
        # -------------------------------------------------

        self._logger.addHandler(console_handler)
        self._logger.addHandler(file_handler)

    # -----------------------------------------------------
    # Internal logging method
    # -----------------------------------------------------

    def _log(
        self,
        level: int,
        message: str,
        *,
        event: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        component: str | None = None,
        **fields: Any,
    ) -> None:
        extra: dict[str, Any] = {}

        request_id = get_request_id()

        if request_id is None:
            raise RuntimeError("request_id is not set in observability context")

        extra["request_id"] = request_id

        if event is not None:
            extra["event"] = event

        if trace_id is not None:
            extra["trace_id"] = trace_id

        if span_id is not None:
            extra["span_id"] = span_id

        if component is not None:
            extra["component"] = component

        # Only add fields that are explicitly supplied.
        extra.update(fields)

        self._logger.log(
            level,
            message,
            extra=extra,
        )

    # -----------------------------------------------------
    # Public methods
    # -----------------------------------------------------

    def debug(self, message: str, **fields: Any) -> None:
        self._log(
            logging.DEBUG,
            message,
            **fields,
        )

    def info(self, message: str, **fields: Any) -> None:
        self._log(
            logging.INFO,
            message,
            **fields,
        )

    def warning(self, message: str, **fields: Any) -> None:
        self._log(
            logging.WARNING,
            message,
            **fields,
        )

    def error(self, message: str, **fields: Any) -> None:
        self._log(
            logging.ERROR,
            message,
            **fields,
        )

    def exception(self, message: str, **fields: Any) -> None:
        request_id = get_request_id()

        if request_id is None:
            raise RuntimeError("request_id is not set in observability context")

        extra = {
            "request_id": request_id,
            **fields,
        }

        self._logger.exception(
            message,
            extra=extra,
        )


# ---------------------------------------------------------
# Global logger instance
# ---------------------------------------------------------

logger = ObservabilityLogger()
