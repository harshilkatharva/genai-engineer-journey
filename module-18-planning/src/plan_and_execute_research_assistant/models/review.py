"""Human plan review models."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from .enums import ReviewAction
from .plan import Plan


class PlanReviewRequest(BaseModel):
    """Pending review request presenting a generated plan to human operator."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the review request",
    )
    plan: Plan = Field(description="Plan submitted for review")
    goal: str = Field(description="Goal for which the plan was generated")
    user_id: str = Field(description="User identifier associated with this plan")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when review was requested",
    )


class PlanReviewDecision(BaseModel):
    """Operator decision submitted in response to a PlanReviewRequest."""

    action: ReviewAction = Field(description="Action taken: approve, modify, or reject")
    reviewer_id: str = Field(description="Identifier of the operator/reviewer submitting decision")
    reason: str | None = Field(
        default=None, description="Optional reasoning, critique, or rejection details"
    )
    modified_plan: Plan | None = Field(
        default=None, description="Modified Plan instance if action is ReviewAction.MODIFY"
    )
    decided_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when decision was made",
    )
