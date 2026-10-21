"""Structured JSON transcript generation and run-level telemetry recorder."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from genai_core.observability import redact_data


@dataclass
class TranscriptStep:
    """Individual execution step (LLM generation, tool action, or HITL approval event)."""

    step_number: int
    step_type: str  # "llm_completion" | "tool_execution" | "approval_pause" | "approval_decision"
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    output: Any | None = None
    status: str = "success"
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert step to dictionary with sensitive data redacted."""
        raw = asdict(self)
        if raw.get("arguments"):
            raw["arguments"] = redact_data(raw["arguments"])
        if raw.get("output") and isinstance(raw["output"], (dict, list)):
            raw["output"] = redact_data(raw["output"])
        return raw


@dataclass
class AgentTranscript:
    """Complete serialized transcript of an agent run."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal: str = ""
    user_id: str | None = None
    status: str = "running"
    termination_reason: str | None = None
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    total_duration_ms: float = 0.0
    total_steps: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    loop_detected: bool = False
    budget_exhausted: bool = False
    approval_required: bool = False
    steps: list[TranscriptStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert full transcript to dictionary with sanitized step logs."""
        data = asdict(self)
        data["steps"] = [step.to_dict() for step in self.steps]
        return data

    def to_json(self, indent: int = 2) -> str:
        """Serialize transcript to JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, output_dir: Path | str) -> Path:
        """Persist transcript JSON file to disk."""
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"run_{self.run_id}.json"
        with file_path.open("w", encoding="utf-8") as f:
            f.write(self.to_json())
        return file_path


class TranscriptRecorder:
    """Telemetry collector building an AgentTranscript across the agent run lifecycle."""

    def __init__(self, goal: str, user_id: str | None = None) -> None:
        self.transcript = AgentTranscript(goal=goal, user_id=user_id)
        self._start_time = time.perf_counter()
        self._current_step_index = 0

    def record_step(
        self,
        step_type: str,
        duration_ms: float = 0.0,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: float | None = None,
        tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
        output: Any | None = None,
        status: str = "success",
        error: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> TranscriptStep:
        """Record an execution step and update run-level aggregations."""
        self._current_step_index += 1
        step = TranscriptStep(
            step_number=self._current_step_index,
            step_type=step_type,
            duration_ms=round(duration_ms, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost_usd, 8) if cost_usd is not None else None,
            tool_name=tool_name,
            arguments=arguments,
            output=output,
            status=status,
            error=error,
            details=details or {},
        )
        self.transcript.steps.append(step)

        # Update totals
        self.transcript.total_steps = len(self.transcript.steps)
        if input_tokens is not None:
            self.transcript.total_input_tokens += max(0, input_tokens)
        if output_tokens is not None:
            self.transcript.total_output_tokens += max(0, output_tokens)
        if cost_usd is not None:
            self.transcript.total_cost_usd = round(
                self.transcript.total_cost_usd + max(0.0, cost_usd), 8
            )

        return step

    def finalize(
        self,
        status: str,
        termination_reason: str | None = None,
        loop_detected: bool = False,
        budget_exhausted: bool = False,
        approval_required: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> AgentTranscript:
        """Finalize the transcript with end time, status, and summary metrics."""
        self.transcript.completed_at = time.time()
        self.transcript.total_duration_ms = round(
            (time.perf_counter() - self._start_time) * 1000.0, 2
        )
        self.transcript.status = status
        self.transcript.termination_reason = termination_reason
        self.transcript.loop_detected = loop_detected
        self.transcript.budget_exhausted = budget_exhausted
        self.transcript.approval_required = approval_required
        if metadata:
            self.transcript.metadata.update(metadata)
        return self.transcript
