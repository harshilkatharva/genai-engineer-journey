from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import BaseModel

from genai_core.llm import GoogleProvider
from genai_core.models import LLMManagerRequest
from genai_core.observability import (
    ObservabilityConfig,
    ObservabilityEvent,
    close_observability,
    configure_observability,
    observability_context,
)
from genai_core.services import LLMService


class IntegrationCityInfo(BaseModel):
    city: str
    country: str


@pytest.fixture(autouse=True)
def cleanup_observability() -> Iterator[None]:
    yield
    close_observability()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_observability_with_caller_log_file(tmp_path: Path) -> None:
    log_dest = tmp_path / "caller_app_logs" / "live_execution.log"
    configure_observability(
        ObservabilityConfig(
            log_file=log_dest,
            log_format="json",
            log_level="DEBUG",
            console=False,
            redact_secrets=True,
        )
    )

    try:
        provider = GoogleProvider()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Google provider init skipped: {exc}")

    service = LLMService(providers={"google": provider}, default_provider="google")

    with observability_context("integration-test-live-trace"):
        req = LLMManagerRequest(
            prompt="Return Paris in France in JSON schema.",
            response_schema=IntegrationCityInfo,
        )
        try:
            response = await service.complete(req)
        except Exception as exc:
            # Skip if API billing or demand quota issue
            if any(term in str(exc).lower() for term in ["quota", "credits", "503"]):
                pytest.skip(f"Live Google API quota/demand limit: {exc}")
            raise

    close_observability()

    if response.error and any(
        term in response.error.message.lower() for term in ["quota", "credits", "503"]
    ):
        pytest.skip(f"Live Google API response error: {response.error.message}")

    assert response.structured_response is not None
    assert log_dest.exists()

    raw_lines = log_dest.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in raw_lines if line.strip()]

    # Ensure start event logged
    start_rec = next(r for r in records if r.get("event") == ObservabilityEvent.LLM_REQUEST_START)
    assert start_rec["provider"] == "google"
    assert start_rec["request_id"] == "integration-test-live-trace"

    # Ensure success event logged with real usage & latency
    success_rec = next(
        r for r in records if r.get("event") == ObservabilityEvent.LLM_REQUEST_SUCCESS
    )
    assert success_rec["provider"] == "google"
    assert success_rec["request_id"] == "integration-test-live-trace"
    assert success_rec["latency_ms"] > 0
    assert success_rec["input_tokens"] > 0
    assert success_rec["output_tokens"] > 0

    # Ensure no raw secret credentials leaked in log file content
    log_content = log_dest.read_text(encoding="utf-8")
    assert "AIza" not in log_content
