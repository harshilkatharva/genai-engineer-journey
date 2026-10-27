"""Exceptions raised during plan validation and structural DAG analysis."""


class PlanValidationError(Exception):
    """Base exception for all plan structural validation errors."""


class EmptyPlanError(PlanValidationError):
    """Raised when a plan contains zero steps."""


class DuplicateStepIdError(PlanValidationError):
    """Raised when two or more steps share the same identifier."""


class MissingDependencyError(PlanValidationError):
    """Raised when a step depends on an ID that does not exist in the plan."""


class SelfDependencyError(PlanValidationError):
    """Raised when a step lists itself as a dependency."""


class CircularDependencyError(PlanValidationError):
    """Raised when a dependency cycle is detected (plan is not a valid DAG)."""


class UnreachablePlanError(PlanValidationError):
    """Raised when a plan contains no entry step with empty dependencies to begin execution."""
