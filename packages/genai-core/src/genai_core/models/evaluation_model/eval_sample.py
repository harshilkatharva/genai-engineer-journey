from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExpectedEvidence(BaseModel):
    """Ground-truth retrieval expectations (supporting both required and alternative chunks)."""

    required: list[str] = Field(default_factory=list)
    any_of: list[list[str]] = Field(default_factory=list)


class ExpectedAnswer(BaseModel):
    """Ground-truth answer expectations."""

    required: list[str] = Field(default_factory=list)
    any_of: list[str] = Field(default_factory=list)
    reference: str | None = None


class EvalSample(BaseModel):
    """Single evaluation test case."""

    id: str | int
    query: str
    context: list[str] | None = None
    expected_evidence: ExpectedEvidence | None = None
    expected_answer: ExpectedAnswer | None = None
    expected_tools: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
