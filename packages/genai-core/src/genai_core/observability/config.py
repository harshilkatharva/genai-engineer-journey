from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ObservabilityConfig(BaseModel):
    """Configuration for genai-core logging and observability.

    Allows consuming applications to configure log destinations, formatting,
    levels, and secret redaction without genai-core imposing hardcoded paths.
    """

    log_level: str | int = "INFO"
    log_file: Path | str | None = Field(
        default=None,
        description=(
            "Destination file path explicitly configured by the caller. "
            "If None, file logging is disabled and no filesystem artifacts are created."
        ),
    )
    log_format: Literal["json", "text"] = Field(
        default="json",
        description="Format of log output: structured 'json' or human-readable 'text'.",
    )
    console: bool = Field(
        default=True,
        description="Whether to emit log records to stdout.",
    )
    redact_secrets: bool = Field(
        default=True,
        description="Whether to scrub API keys, tokens, and credentials automatically.",
    )
    extra_redaction_patterns: list[str] = Field(
        default_factory=list,
        description="Optional custom regex patterns for secret masking.",
    )
    logger_name: str = Field(
        default="genai_core",
        description="Root logger name for genai-core components.",
    )
    rotate_logs: bool = Field(
        default=True,
        description="Whether to use RotatingFileHandler when log_file is specified.",
    )
    max_bytes: int = Field(
        default=10 * 1024 * 1024,
        description="Maximum file size in bytes before rotating (default 10 MB).",
    )
    backup_count: int = Field(
        default=3,
        description="Number of backup log files to retain upon rotation.",
    )

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "ignore",
    }
