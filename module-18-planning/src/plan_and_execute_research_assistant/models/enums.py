"""Domain enums for Plan-and-Execute Research Assistant."""

from enum import Enum


class StepStatus(str, Enum):
    """Execution lifecycle status of an individual plan step."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class RunStatus(str, Enum):
    """End-to-end execution lifecycle status of an assistant run."""

    INITIALIZING = "initializing"
    PLANNING = "planning"
    REVIEW_PENDING = "review_pending"
    EXECUTING = "executing"
    REPLANNING = "replanning"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReviewAction(str, Enum):
    """Human plan review decision action."""

    APPROVE = "approve"
    MODIFY = "modify"
    REJECT = "reject"
