from __future__ import annotations

import json
import time
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..models.llm_model.llm_response_model import ChatMessage


class RiskLevel(StrEnum):
    """Classification of tool operational risk."""

    READ_ONLY = "read_only"
    REVERSIBLE_WRITE = "reversible_write"
    IRREVERSIBLE = "irreversible"


class ToolExecutionStatus(StrEnum):
    """Execution status for a single tool call."""

    SUCCESS = "success"
    VALIDATION_ERROR = "validation_error"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT = "timeout"
    APPROVAL_REQUIRED = "approval_required"
    REJECTED = "rejected"


class ToolResult(BaseModel):
    """Standard outcome of a tool execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    call_id: str
    tool_name: str
    status: ToolExecutionStatus
    content: Any | None = None
    error_message: str | None = None
    latency_ms: float = 0.0
    truncated: bool = False

    def is_success(self) -> bool:
        """Return True if execution completed successfully."""
        return self.status == ToolExecutionStatus.SUCCESS

    def to_chat_message(self) -> ChatMessage:
        """Convert result into a tool-role ChatMessage suitable for LLM context."""
        if self.error_message:
            text = f"Error: {self.error_message}"
        elif self.content is None:
            text = ""
        elif isinstance(self.content, str):
            text = self.content
        elif isinstance(self.content, BaseModel):
            text = self.content.model_dump_json()
        else:
            try:
                text = json.dumps(self.content, default=str)
            except (TypeError, ValueError):
                text = str(self.content)

        return ChatMessage(
            role="tool",
            content=text,
            tool_call_id=self.call_id,
            tool_name=self.tool_name,
        )


class ApprovalRequest(BaseModel):
    """Encapsulates a pending authorization request for a risky tool operation."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.IRREVERSIBLE
    reason: str = "High risk operation requires approval"
    created_at: float = Field(default_factory=time.time)


class ApprovalDecision(BaseModel):
    """The caller or human-in-the-loop decision for a pending approval request."""

    approved: bool
    reason: str | None = None
    modified_arguments: dict[str, Any] | None = None


class ToolBudget(BaseModel):
    """Resource constraints and accounting for tool execution loops."""

    max_steps: int = 10
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    deadline_monotonic: float | None = None

    current_steps: int = 0
    current_tokens: int = 0
    current_cost_usd: float = 0.0

    def is_deadline_exceeded(self) -> bool:
        """Check if monotonic deadline has passed."""
        if self.deadline_monotonic is not None:
            return time.monotonic() >= self.deadline_monotonic
        return False

    def is_step_budget_exhausted(self) -> bool:
        """Check if max execution steps have been reached."""
        return self.current_steps >= self.max_steps

    def is_token_budget_exhausted(self) -> bool:
        """Check if token budget has been exceeded."""
        if self.max_tokens is not None:
            return self.current_tokens >= self.max_tokens
        return False

    def is_cost_budget_exhausted(self) -> bool:
        """Check if cost budget has been exceeded."""
        if self.max_cost_usd is not None:
            return self.current_cost_usd >= self.max_cost_usd
        return False

    def can_step(self) -> bool:
        """Return True if loop is permitted to execute another step."""
        if self.is_step_budget_exhausted():
            return False
        if self.is_deadline_exceeded():
            return False
        if self.is_token_budget_exhausted():
            return False
        return not self.is_cost_budget_exhausted()

    def record_step(self, tokens: int = 0, cost_usd: float = 0.0) -> None:
        """Record resource consumption for a completed loop iteration."""
        self.current_steps += 1
        self.current_tokens += max(0, tokens)
        self.current_cost_usd += max(0.0, cost_usd)


class ToolLoopStatus(StrEnum):
    """Terminal or pause state of a tool loop execution."""

    COMPLETED = "completed"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    BUDGET_EXCEEDED = "budget_exceeded"
    APPROVAL_REQUIRED = "approval_required"
    FAILED = "failed"


class ToolLoopResult(BaseModel):
    """Comprehensive result returned upon tool loop termination or suspension."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: ToolLoopStatus
    final_text: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    iterations: int = 0
    total_tool_calls: int = 0
    pending_approval: ApprovalRequest | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
