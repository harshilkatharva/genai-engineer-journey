from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from customer_support_agent.core import get_settings

from .context import get_request_id


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "request_id": str(getattr(record, "request_id", None)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in (
            "event",
            "component",
            "provider",
            "model",
            "tool_name",
            "arguments",
            "success",
            "duration_ms",
            "iteration",
            "error_type",
            "error_message",
            "input_tokens",
            "output_tokens",
            "query",
            "row_count",
        ):
            if hasattr(record, key):
                data[key] = getattr(record, key)
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data, default=str, ensure_ascii=False)


class ObservabilityLogger:
    def __init__(self, name: str = "customer_support_agent") -> None:
        self._logger = logging.getLogger(name)
        self._logger.setLevel(get_settings().log_level)
        self._logger.propagate = False
        if self._logger.handlers:
            return
        formatter = JsonFormatter()
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        log_file = Path(get_settings().tool_call_log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        self._logger.addHandler(console)
        self._logger.addHandler(file_handler)

    def _log(self, level: int, message: str, **fields: Any) -> None:
        fields["request_id"] = get_request_id()
        self._logger.log(level, message, extra=fields)

    def info(self, message: str, **fields: Any) -> None:
        self._log(logging.INFO, message, **fields)

    def warning(self, message: str, **fields: Any) -> None:
        self._log(logging.WARNING, message, **fields)

    def error(self, message: str, **fields: Any) -> None:
        self._log(logging.ERROR, message, **fields)

    def exception(self, message: str, **fields: Any) -> None:
        fields["request_id"] = get_request_id()
        self._logger.exception(message, extra=fields)


logger = ObservabilityLogger()
