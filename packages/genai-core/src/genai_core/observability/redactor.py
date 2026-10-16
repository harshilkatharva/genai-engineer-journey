from __future__ import annotations

import logging
import re
from typing import Any

# Common API key and credential regex patterns
DEFAULT_SECRET_PATTERNS: list[re.Pattern[str]] = [
    # Google API Key (typically starts with AIzaSy... 35-40 chars total)
    re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
    # OpenAI API Key (standard sk-..., sk-proj-...)
    re.compile(r"sk-[a-zA-Z0-9_\-]{20,}"),
    # Anthropic API Key (sk-ant-...)
    re.compile(r"sk-ant-[a-zA-Z0-9_\-]{20,}"),
    # Bearer tokens in headers or strings
    re.compile(r"(Bearer\s+)[a-zA-Z0-9._\-]{10,}", re.IGNORECASE),
    # Key-value secret assignments like api_key=xyz or "secret": "xyz"
    re.compile(
        r"""(?i)(api[_-]?key|access[_-]?token|secret|password|authorization)\s*[:=]\s*['"]?([a-zA-Z0-9_\-\.]{8,})['"]?"""
    ),
]

SENSITIVE_KEY_NAMES = {
    "api_key",
    "apikey",
    "secret",
    "client_secret",
    "password",
    "token",
    "access_token",
    "refresh_token",
    "bearer_token",
    "auth_token",
    "id_token",
    "auth",
    "authorization",
    "google_api_key",
    "openai_api_key",
    "anthropic_api_key",
    "secret_key",
    "private_key",
    "access_key",
}

NON_SENSITIVE_METRIC_KEYS = {
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "token_count",
    "prompt_tokens",
    "completion_tokens",
    "tokens",
}

CREDENTIAL_SUFFIXES = (
    "api_key",
    "apikey",
    "secret_key",
    "private_key",
    "access_key",
    "_token",
    "_secret",
    "_password",
)

REDACTED_SUBSTITUTE = "[REDACTED]"


def is_sensitive_key(key: str) -> bool:
    """Determine if a dictionary key represents credentials or secret tokens."""
    str_key = str(key).lower()
    if str_key in NON_SENSITIVE_METRIC_KEYS:
        return False
    if str_key in SENSITIVE_KEY_NAMES:
        return True
    return any(str_key.endswith(suffix) for suffix in CREDENTIAL_SUFFIXES)


def redact_string(
    text: str,
    patterns: list[re.Pattern[str]] | None = None,
) -> str:
    """Redact known secret patterns from a string."""
    if not text:
        return text

    compiled_patterns = patterns or DEFAULT_SECRET_PATTERNS
    result = text
    for pattern in compiled_patterns:
        # If pattern has capture groups (like key-value or bearer), keep group 1 prefix if helpful
        if pattern.groups == 2:
            result = pattern.sub(rf"\1={REDACTED_SUBSTITUTE}", result)
        elif pattern.groups == 1:
            result = pattern.sub(rf"\1{REDACTED_SUBSTITUTE}", result)
        else:
            result = pattern.sub(REDACTED_SUBSTITUTE, result)
    return result


def redact_data(
    data: Any,
    patterns: list[re.Pattern[str]] | None = None,
) -> Any:
    """Recursively redact secrets in strings, dictionaries, lists, and tuples."""
    if isinstance(data, str):
        return redact_string(data, patterns)
    if isinstance(data, dict):
        redacted_dict: dict[str, Any] = {}
        for key, value in data.items():
            if is_sensitive_key(str(key)):
                redacted_dict[key] = REDACTED_SUBSTITUTE
            else:
                redacted_dict[key] = redact_data(value, patterns)
        return redacted_dict
    if isinstance(data, list):
        return [redact_data(item, patterns) for item in data]
    if isinstance(data, tuple):
        return tuple(redact_data(item, patterns) for item in data)
    return data


class RedactingFilter(logging.Filter):
    """Logging filter that scrubs credentials and sensitive tokens from log records."""

    def __init__(
        self,
        name: str = "",
        extra_patterns: list[str] | None = None,
    ) -> None:
        super().__init__(name)
        self.patterns = list(DEFAULT_SECRET_PATTERNS)
        if extra_patterns:
            for pat in extra_patterns:
                self.patterns.append(re.compile(pat))

    def filter(self, record: logging.LogRecord) -> bool:
        # Scrub message
        if isinstance(record.msg, str):
            record.msg = redact_string(record.msg, self.patterns)

        # Scrub formatting args
        if isinstance(record.args, dict):
            record.args = redact_data(record.args, self.patterns)
        elif isinstance(record.args, tuple):
            record.args = tuple(redact_data(a, self.patterns) for a in record.args)

        # Scrub custom extra attributes on the log record
        standard_attrs = {
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
        for attr, val in list(record.__dict__.items()):
            if attr not in standard_attrs and not attr.startswith("_"):
                setattr(record, attr, redact_data(val, self.patterns))

        return True
