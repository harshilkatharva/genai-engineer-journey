from __future__ import annotations


class ToolError(Exception):
    """Base exception for all tool subsystem operations in genai-core."""


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not found in the registry."""


class ToolValidationError(ToolError):
    """Raised when tool arguments fail validation or are malformed."""


class ToolExecutionError(ToolError):
    """Raised when a tool handler encounters an unexpected runtime error."""


class ToolTimeoutError(ToolError):
    """Raised when tool execution exceeds its configured timeout deadline."""


class ToolApprovalRequiredError(ToolError):
    """Raised when a tool invocation requires human or external approval before running."""


class ToolBudgetExceededError(ToolError):
    """Raised when a tool execution loop exceeds configured step, token, or cost budgets."""


class ToolLoopDetectedError(ToolError):
    """Raised when repeated identical tool calls indicate an infinite loop or thrashing."""
