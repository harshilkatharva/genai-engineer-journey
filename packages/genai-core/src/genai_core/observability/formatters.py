from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from .context import get_request_id
from .redactor import redact_data

STANDARD_RECORD_ATTRS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Formats log records as structured, single-line JSON with dynamic extras."""

    def __init__(self, redact: bool = True) -> None:
        super().__init__()
        self.redact = redact

    def format(self, record: logging.LogRecord) -> str:
        # Resolve request_id from record or active context
        req_id = getattr(record, "request_id", None) or get_request_id()

        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if req_id is not None:
            payload["request_id"] = str(req_id)

        # Include custom fields passed in extra
        for key, val in record.__dict__.items():
            if key not in STANDARD_RECORD_ATTRS and not key.startswith("_") and key != "request_id":
                payload[key] = val

        # Handle exceptions safely
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            payload["exception"] = record.exc_text
            exc_class = record.exc_info[0]
            exc_instance = record.exc_info[1]
            if exc_class and "error_type" not in payload:
                payload["error_type"] = exc_class.__name__
            if exc_instance and "error_message" not in payload:
                payload["error_message"] = str(exc_instance)

        if self.redact:
            payload = redact_data(payload)

        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Formats log records as human-readable text including structured metadata."""

    def __init__(self, redact: bool = True) -> None:
        super().__init__(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.redact = redact

    def format(self, record: logging.LogRecord) -> str:
        base_msg = super().format(record)
        req_id = getattr(record, "request_id", None) or get_request_id()

        # Collect structured contextual fields
        extras: list[str] = []
        if req_id:
            extras.append(f"request_id={req_id}")

        for key, val in record.__dict__.items():
            if key not in STANDARD_RECORD_ATTRS and not key.startswith("_") and key != "request_id":
                extras.append(f"{key}={val}")

        extra_str = f" [{' '.join(extras)}]" if extras else ""
        final_text = f"{base_msg}{extra_str}"
        if self.redact:
            return str(redact_data(final_text))
        return final_text
