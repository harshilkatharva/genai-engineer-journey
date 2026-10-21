"""Trusted authorization validation binding approvals to actors, operations, and arguments."""

from __future__ import annotations

from typing import Any, ClassVar

from genai_core.exceptions import ToolValidationError
from genai_core.tools.models import ApprovalDecision, ApprovalRequest


class AuthorizationValidator:
    """Validates that operator decisions comply with security boundaries."""

    SUPPORT_OPERATIONS: ClassVar[set[str]] = {
        "execute_refund",
        "purge_expired",
        "delete_customer_data",
    }

    USER_MEMORY_OPERATIONS: ClassVar[set[str]] = {
        "delete_longterm_memory_all",
        "delete_longterm_memory_specific",
        "delete_episodic_memory_all",
        "delete_episodic_memory_episodes",
    }

    @classmethod
    def validate_approval(
        cls,
        request: ApprovalRequest,
        decision: ApprovalDecision,
        actor_id: str,
        actor_role: str,
    ) -> bool:
        """Validate that the reviewing actor has the authority to approve this specific action.

        - Memory deletion: Must be confirmed by the specific user who owns the memory.
        - Support actions (refunds, purge, customer data delete): Must be approved by a support-team actor.
        - LLM or automated callers cannot approve their own actions.
        """
        if not decision.approved:
            return True  # Rejections are always acceptable

        if actor_role == "agent" or actor_role == "llm":
            raise ToolValidationError("LLM is prohibited from approving its own actions.")

        tool_name = request.tool_name

        # Policy 1: Memory deletion requires confirmation from the owning user
        if tool_name in cls.USER_MEMORY_OPERATIONS:
            target_user_id = request.arguments.get("user_id")
            if not target_user_id or str(target_user_id).strip() != str(actor_id).strip():
                raise ToolValidationError(
                    f"Memory deletion operation '{tool_name}' must be confirmed by the memory "
                    f"owner ('{target_user_id}'), but got approval from '{actor_id}'."
                )
            return True

        # Policy 2: Support actions require support_team role
        if tool_name in cls.SUPPORT_OPERATIONS:
            if actor_role != "support_team" and actor_role != "admin":
                raise ToolValidationError(
                    f"Operation '{tool_name}' requires authorization from 'support_team', "
                    f"but actor has role '{actor_role}'."
                )
            return True

        return True

    @classmethod
    def revalidate_arguments(
        cls,
        tool_name: str,
        effective_arguments: dict[str, Any],
    ) -> None:
        """Sanity check effective (possibly modified) arguments before execution."""
        if tool_name == "execute_refund":
            amount = effective_arguments.get("amount_usd")
            if amount is None or float(amount) <= 0.0:
                raise ToolValidationError("Refund amount must be a positive number.")
            if not effective_arguments.get("customer_id"):
                raise ToolValidationError("Refund requires a valid customer_id.")

        elif tool_name == "delete_customer_data":
            if not effective_arguments.get("customer_id"):
                raise ToolValidationError("delete_customer_data requires customer_id.")
            if not effective_arguments.get("data_scope"):
                raise ToolValidationError("delete_customer_data requires explicit data_scope.")
