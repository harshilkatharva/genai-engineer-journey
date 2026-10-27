"""Protocols for step execution engines and specialist agent routers."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..models.plan import PlanStep


@runtime_checkable
class StepExecutor(Protocol):
    """Protocol for step execution engines (Module 17 BoundedAgent or Module 20 Specialist Router)."""

    def get_available_capabilities(self) -> dict[str, str]:
        """Return a dictionary of available tool/capability names mapped to descriptions.

        Allows the Planner LLM to dynamically discover available tools without hardcoding.
        """
        ...

    async def execute_step(
        self,
        step: PlanStep,
        predecessor_context: dict[str, str],
        user_id: str,
    ) -> Any:
        """Execute an individual plan step within the executor's bounded loop.

        Args:
            step: The PlanStep to execute.
            predecessor_context: Mapping of predecessor step ID to its verified result summary.
            user_id: User identifier for authorization and memory context.

        Returns:
            Step execution result / AgentRunResult.
        """
        ...
