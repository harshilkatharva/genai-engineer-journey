from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol, runtime_checkable

from .models import ApprovalDecision, ApprovalRequest


class LoopDetector:
    """Detects repeated or cyclical tool invocations to prevent infinite execution loops."""

    def __init__(self, max_repeated_calls: int = 3) -> None:
        self.max_repeated_calls = max_repeated_calls
        self._history: list[str] = []
        self._consecutive_counts: dict[str, int] = {}
        self._last_signature: str | None = None

    def _compute_signature(self, tool_name: str, arguments: Any) -> str:
        """Create a deterministic hash signature from tool name and arguments."""
        if isinstance(arguments, str):
            try:
                norm_args = json.loads(arguments)
            except (json.JSONDecodeError, TypeError, ValueError):
                norm_args = arguments
        elif hasattr(arguments, "model_dump"):
            norm_args = arguments.model_dump()
        else:
            norm_args = arguments

        try:
            serialized = json.dumps(norm_args, sort_keys=True, default=str)
        except (TypeError, ValueError):
            serialized = str(norm_args)

        content = f"{tool_name}:{serialized}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def record_and_check(self, tool_name: str, arguments: Any) -> bool:
        """Record an invocation and return True if an infinite loop pattern is detected.

        Triggers when the exact same tool and arguments are invoked consecutively
        max_repeated_calls times.
        """
        signature = self._compute_signature(tool_name, arguments)
        self._history.append(signature)

        if signature == self._last_signature:
            count = self._consecutive_counts.get(signature, 1) + 1
            self._consecutive_counts[signature] = count
        else:
            self._last_signature = signature
            count = 1
            self._consecutive_counts[signature] = 1

        return count >= self.max_repeated_calls

    def reset(self) -> None:
        """Reset the loop detection tracking state."""
        self._history.clear()
        self._consecutive_counts.clear()
        self._last_signature = None


@runtime_checkable
class ApprovalStore(Protocol):
    """Protocol for persisting and managing human-in-the-loop tool approval requests."""

    async def save_request(self, request: ApprovalRequest) -> None:
        """Persist a new pending approval request."""
        ...

    async def get_request(self, request_id: str) -> ApprovalRequest | None:
        """Retrieve an approval request by ID."""
        ...

    async def record_decision(self, request_id: str, decision: ApprovalDecision) -> None:
        """Record an approval or rejection decision."""
        ...

    async def get_decision(self, request_id: str) -> ApprovalDecision | None:
        """Retrieve the recorded decision for a request, if decided."""
        ...

    async def list_pending_requests(self) -> list[ApprovalRequest]:
        """List all requests awaiting decision."""
        ...


class InMemoryApprovalStore(ApprovalStore):
    """In-memory reference implementation of ApprovalStore."""

    def __init__(self) -> None:
        self._requests: dict[str, ApprovalRequest] = {}
        self._decisions: dict[str, ApprovalDecision] = {}

    async def save_request(self, request: ApprovalRequest) -> None:
        self._requests[request.request_id] = request

    async def get_request(self, request_id: str) -> ApprovalRequest | None:
        return self._requests.get(request_id)

    async def record_decision(self, request_id: str, decision: ApprovalDecision) -> None:
        if request_id not in self._requests:
            raise KeyError(f"Approval request '{request_id}' not found.")
        self._decisions[request_id] = decision

    async def get_decision(self, request_id: str) -> ApprovalDecision | None:
        return self._decisions.get(request_id)

    async def list_pending_requests(self) -> list[ApprovalRequest]:
        return [req for req_id, req in self._requests.items() if req_id not in self._decisions]
