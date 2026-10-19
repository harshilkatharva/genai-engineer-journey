from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from genai_core.exceptions import LLMValidationError
from genai_core.models import (
    ChatMessage,
    LLMManagerRequest,
    LLMResponseModel,
)
from genai_core.observability import (
    JsonFormatter,
    ObservabilityConfig,
    ObservabilityEvent,
    RedactingFilter,
    TextFormatter,
    close_observability,
    configure_observability,
    get_logger,
    get_request_id,
    observability_context,
    redact_data,
    redact_string,
    reset_request_id,
    safe_log,
    set_request_id,
)
from genai_core.services import LLMService


@pytest.fixture(autouse=True)
def cleanup_logger() -> Iterator[None]:
    yield
    close_observability()


class DummySchema(BaseModel):
    summary: str


# -------------------------------------------------------------------------
# 1. Configuration & Destination Tests
# -------------------------------------------------------------------------


def test_observability_config_defaults() -> None:
    config = ObservabilityConfig()
    assert config.log_level == "INFO"
    assert config.log_file is None
    assert config.log_format == "json"
    assert config.console is True
    assert config.redact_secrets is True


def test_configure_observability_custom_file_destination(tmp_path: Path) -> None:
    custom_log = tmp_path / "app_logs" / "nested" / "service.log"
    config = ObservabilityConfig(
        log_file=custom_log,
        log_level="DEBUG",
        log_format="json",
        console=False,
    )
    logger = configure_observability(config)

    assert custom_log.parent.exists()
    assert logger.level == logging.DEBUG

    logger.info("Test log entry to caller destination", extra={"caller_key": "val123"})
    close_observability()

    assert custom_log.exists()
    content = custom_log.read_text(encoding="utf-8").strip()
    data = json.loads(content)
    assert data["message"] == "Test log entry to caller destination"
    assert data["caller_key"] == "val123"
    assert data["level"] == "INFO"
    assert "timestamp" in data


def test_configure_observability_text_format(tmp_path: Path) -> None:
    log_file = tmp_path / "text.log"
    config = ObservabilityConfig(
        log_file=log_file,
        log_format="text",
        console=False,
    )
    logger = configure_observability(config)
    logger.info("Human readable test", extra={"component": "test_core"})
    close_observability()

    content = log_file.read_text(encoding="utf-8")
    assert "[INFO]" in content
    assert "Human readable test" in content
    assert "component=test_core" in content


def test_text_formatter_direct() -> None:
    formatter = TextFormatter(redact=True)
    record = logging.LogRecord(
        name="test_text",
        level=logging.WARNING,
        pathname="test.py",
        lineno=1,
        msg="Warning with secret AIzaSyD-1234567890abcdefghijklmnopqr",
        args=(),
        exc_info=None,
    )
    record.request_id = "req-text-1"  # type: ignore[attr-defined]
    out = formatter.format(record)
    assert "[WARNING]" in out
    assert "[REDACTED]" in out
    assert "request_id=req-text-1" in out


def test_configure_observability_reconfiguration_idempotent(tmp_path: Path) -> None:
    log_file = tmp_path / "reconfig.log"
    config = ObservabilityConfig(log_file=log_file, console=False)

    # Configure once
    logger = configure_observability(config)
    logger.info("Message 1")

    # Reconfigure
    configure_observability(config)
    logger.info("Message 2")
    close_observability()

    lines = [line for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Exactly 2 lines (no duplicate handlers emitting twice)
    assert len(lines) == 2


def test_get_logger_default_null_handler() -> None:
    close_observability()
    logger = get_logger("genai_core.unconfigured_sub")
    assert logger.name == "genai_core.unconfigured_sub"
    # Should log without error or warning even without configuration
    logger.info("Silent log before configuration")


# -------------------------------------------------------------------------
# 2. Context & Request ID Tests
# -------------------------------------------------------------------------


def test_request_id_context_vars() -> None:
    # No request ID by default - must return None without raising
    assert get_request_id() is None

    # Set explicit request ID
    token = set_request_id("req-abc-123")
    assert get_request_id() == "req-abc-123"
    reset_request_id(token)
    assert get_request_id() is None

    # Auto-generation when None provided
    token2 = set_request_id(None)
    auto_id = get_request_id()
    assert auto_id is not None
    assert len(auto_id) == 36  # UUID4 format
    reset_request_id(token2)


def test_observability_context_manager() -> None:
    assert get_request_id() is None
    with observability_context("scoped-uuid-99") as active_id:
        assert active_id == "scoped-uuid-99"
        assert get_request_id() == "scoped-uuid-99"
    assert get_request_id() is None


# -------------------------------------------------------------------------
# 3. Security & Redaction Tests
# -------------------------------------------------------------------------


def test_redact_string_patterns() -> None:
    # Google API Key
    google_key = "AIzaSyD-1234567890abcdefghijklmnopqr"
    assert redact_string(f"Key is {google_key}") == "Key is [REDACTED]"

    # OpenAI API Key
    openai_key = "sk-12345678901234567890abcdefg"
    assert redact_string(f"OpenAI: {openai_key}") == "OpenAI: [REDACTED]"

    # Anthropic API Key
    claude_key = "sk-ant-12345678901234567890abcdef"
    assert redact_string(f"Claude: {claude_key}") == "Claude: [REDACTED]"

    # Bearer token
    assert (
        redact_string("Authorization: Bearer secrettoken1234567890")
        == "Authorization: Bearer [REDACTED]"
    )


def test_redact_data_nested_dictionary() -> None:
    payload = {
        "user": "alice",
        "api_key": "raw_sensitive_key",
        "nested": {
            "token": "sensitive_jwt_token",
            "safe_val": 42,
            "google_api_key": "AIzaSyD-1234567890abcdefghijklmnopqr",
        },
        "items": [
            "sk-12345678901234567890abcdefg",
            {"password": "mypassword123", "normal": "ok"},
        ],
    }
    redacted = redact_data(payload)
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["nested"]["token"] == "[REDACTED]"
    assert redacted["nested"]["google_api_key"] == "[REDACTED]"
    assert redacted["nested"]["safe_val"] == 42
    assert redacted["items"][0] == "[REDACTED]"
    assert redacted["items"][1]["password"] == "[REDACTED]"
    assert redacted["items"][1]["normal"] == "ok"


def test_json_formatter_redaction_and_exceptions() -> None:
    formatter = JsonFormatter(redact=True)
    record = logging.LogRecord(
        name="test_logger",
        level=logging.ERROR,
        pathname="test.py",
        lineno=10,
        msg="Error connecting with key AIzaSyD-1234567890abcdefghijklmnopqr",
        args=(),
        exc_info=None,
    )
    record.api_key = "sk-12345678901234567890abcdefg"  # type: ignore[attr-defined]

    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["message"] == "Error connecting with key [REDACTED]"
    assert parsed["api_key"] == "[REDACTED]"


def test_redacting_filter_on_log_record() -> None:
    filter_obj = RedactingFilter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Using key %s",
        args=("AIzaSyD-1234567890abcdefghijklmnopqr",),
        exc_info=None,
    )
    record.secret_token = "sk-12345678901234567890abcdefg"  # type: ignore[attr-defined]
    filter_obj.filter(record)

    assert record.args == ("[REDACTED]",)
    assert record.secret_token == "[REDACTED]"  # type: ignore[attr-defined]


# -------------------------------------------------------------------------
# 4. Resilience & Safe Logging Tests
# -------------------------------------------------------------------------


def test_safe_log_never_raises() -> None:
    mock_logger = MagicMock(spec=logging.Logger)
    mock_logger.log.side_effect = RuntimeError("Disk full / handler error")

    # safe_log must catch and suppress without bubbling up
    safe_log(mock_logger, logging.INFO, "Should not raise", key="value")
    mock_logger.log.assert_called_once()


# -------------------------------------------------------------------------
# 5. LLMService Instrumentation Tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_service_observability_success(tmp_path: Path) -> None:
    log_file = tmp_path / "llm_success.log"
    configure_observability(
        ObservabilityConfig(log_file=log_file, log_format="json", console=False)
    )

    mock_provider = MagicMock()
    mock_provider.complete_structure = AsyncMock(
        return_value=LLMResponseModel(
            text="response text",
            model="gemini-3.5-flash-lite",
            structured_response=DummySchema(summary="done"),
            input_tokens=15,
            output_tokens=30,
            latency_ms=15.0,
        )
    )
    service = LLMService(providers={"google": mock_provider}, default_provider="google")

    with observability_context("test-trace-id-123"):
        request = LLMManagerRequest(
            prompt="Hello world",
            response_schema=DummySchema,
        )
        response = await service.complete(request)

    close_observability()
    assert response.structured_response is not None

    lines = [
        json.loads(line)
        for line in log_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) >= 2

    # Verify Start event
    start_event = next(e for e in lines if e.get("event") == ObservabilityEvent.LLM_REQUEST_START)
    assert start_event["provider"] == "google"
    assert start_event["request_id"] == "test-trace-id-123"
    assert start_event["has_schema"] is True

    # Verify Success event
    success_event = next(
        e for e in lines if e.get("event") == ObservabilityEvent.LLM_REQUEST_SUCCESS
    )
    assert success_event["provider"] == "google"
    assert success_event["model"] == "gemini-3.5-flash-lite"
    assert success_event["input_tokens"] == 15
    assert success_event["output_tokens"] == 30
    assert success_event["request_id"] == "test-trace-id-123"
    assert "latency_ms" in success_event


@pytest.mark.asyncio
async def test_llm_service_observability_validation_error(tmp_path: Path) -> None:
    log_file = tmp_path / "llm_val_error.log"
    configure_observability(
        ObservabilityConfig(log_file=log_file, log_format="json", console=False)
    )

    mock_provider = MagicMock()
    mock_provider.complete_structure = AsyncMock(
        side_effect=LLMValidationError("Invalid schema parsed from response")
    )
    service = LLMService(providers={"google": mock_provider}, default_provider="google")

    request = LLMManagerRequest(prompt="Test", response_schema=DummySchema)
    response = await service.complete(request)

    close_observability()
    assert response.error is not None
    assert response.error.code == "validation_error"

    lines = [
        json.loads(line)
        for line in log_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failed_event = next(e for e in lines if e.get("event") == ObservabilityEvent.LLM_REQUEST_FAILED)
    assert failed_event["provider"] == "google"
    assert failed_event["error_code"] == "validation_error"
    assert failed_event["retryable"] is False


@pytest.mark.asyncio
async def test_llm_service_observability_unsupported_provider(tmp_path: Path) -> None:
    log_file = tmp_path / "llm_unsupported.log"
    configure_observability(
        ObservabilityConfig(log_file=log_file, log_format="json", console=False)
    )

    service = LLMService(providers={}, default_provider="unknown_llm")
    request = LLMManagerRequest(prompt="Test")
    response = await service.complete(request)

    close_observability()
    assert response.error is not None
    assert response.error.code == "unsupported_provider"

    lines = [
        json.loads(line)
        for line in log_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failed_event = next(e for e in lines if e.get("event") == ObservabilityEvent.LLM_REQUEST_FAILED)
    assert failed_event["error_code"] == "unsupported_provider"


@pytest.mark.asyncio
async def test_llm_service_stream_observability(tmp_path: Path) -> None:
    log_file = tmp_path / "llm_stream.log"
    configure_observability(
        ObservabilityConfig(log_file=log_file, log_format="json", console=False)
    )

    async def mock_stream_gen(messages: list[ChatMessage]):
        yield "chunk1 "
        yield "chunk2"

    mock_provider = MagicMock()
    mock_provider.stream = mock_stream_gen

    service = LLMService(providers={"google": mock_provider})
    tokens = []
    async for tok in service.stream("google", "hello"):
        tokens.append(tok)

    close_observability()
    assert "".join(tokens) == "chunk1 chunk2"

    lines = [
        json.loads(line)
        for line in log_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    start_ev = next(e for e in lines if e.get("event") == ObservabilityEvent.LLM_STREAM_START)
    success_ev = next(e for e in lines if e.get("event") == ObservabilityEvent.LLM_STREAM_SUCCESS)
    assert start_ev["provider"] == "google"
    assert success_ev["provider"] == "google"
    assert "latency_ms" in success_ev
