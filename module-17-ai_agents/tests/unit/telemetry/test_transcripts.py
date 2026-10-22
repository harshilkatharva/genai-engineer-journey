"""Unit tests for structured JSON transcripts and redaction."""

from __future__ import annotations

import json
from pathlib import Path

from research_and_support_agent.telemetry.transcript import TranscriptRecorder


def test_transcript_generation_and_redaction(tmp_path: Path) -> None:
    """Verify transcript assembly, redaction of sensitive info, and file saving."""
    recorder = TranscriptRecorder(goal="Refund customer C-101", user_id="U-1")

    # Step 1: LLM completion
    recorder.record_step(
        step_type="llm_completion",
        duration_ms=450.0,
        input_tokens=120,
        output_tokens=30,
        cost_usd=0.00002,
        output="Calling execute_refund",
    )

    # Step 2: Sensitive tool action with secret keys/card numbers
    recorder.record_step(
        step_type="tool_execution",
        duration_ms=120.0,
        tool_name="execute_refund",
        arguments={
            "customer_id": "C-101",
            "amount_usd": 50.0,
            "api_key": "sk-secret-token-12345",
            "credit_card": "4111-2222-3333-4444",
        },
        output={"transaction_id": "tx_99", "secret": "internal_key_abc"},
        status="success",
    )

    # Finalize
    transcript = recorder.finalize(status="completed")
    assert transcript.total_steps == 2
    assert transcript.total_input_tokens == 120
    assert transcript.total_output_tokens == 30
    assert transcript.total_cost_usd > 0.0

    # Verify sensitive data redaction
    as_dict = transcript.to_dict()
    step2 = as_dict["steps"][1]
    raw_json = json.dumps(step2)
    assert "sk-secret-token-12345" not in raw_json
    assert "[REDACTED" in raw_json

    # Test file saving
    saved_path = transcript.save(tmp_path)
    assert saved_path.exists()
    assert saved_path.name.startswith("run_")
    with saved_path.open() as f:
        loaded = json.load(f)
        assert loaded["goal"] == "Refund customer C-101"
        assert loaded["status"] == "completed"
