import json
import time
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

# Task 2


class AgentLimits(BaseModel):
    max_steps: int = 5
    token_budget: int = 1000
    deadline_seconds: float = 10.0


class AgentResult(BaseModel):
    status: str  # "completed", "max_steps_exceeded", "token_budget_exceeded", "deadline_exceeded"
    output: str | None = None
    steps_taken: int = 0
    tokens_used: int = 0


TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "get_time": lambda: "2026-10-14T17:00:00",
    "calculator": lambda a, b, operation: a + b,
}


async def run_agent(
    goal: str,
    client: Any,
    model: str = "gemini-3.5-flash-lite",
    limits: AgentLimits | None = None,
) -> AgentResult:
    limits = limits or AgentLimits()
    start_time = time.time()
    steps_taken = 0
    tokens_used = 0
    last_output: str | None = None

    interaction = client.interactions.create(model=model, input=goal)

    while True:
        # 1. Wall-clock deadline check
        if time.time() - start_time >= limits.deadline_seconds:
            return AgentResult(
                status="deadline_exceeded",
                output=last_output,
                steps_taken=steps_taken,
                tokens_used=tokens_used,
            )

        # 2. Step cap check
        if steps_taken >= limits.max_steps:
            return AgentResult(
                status="max_steps_exceeded",
                output=last_output,
                steps_taken=steps_taken,
                tokens_used=tokens_used,
            )

        # 3. Token budget check
        usage = getattr(interaction, "usage", None)
        tokens_used += getattr(usage, "total_tokens", 0) or 0
        if tokens_used > limits.token_budget:
            return AgentResult(
                status="token_budget_exceeded",
                output=last_output,
                steps_taken=steps_taken,
                tokens_used=tokens_used,
            )

        last_output = getattr(interaction, "output_text", None)
        tool_calls = [
            step
            for step in (getattr(interaction, "steps", None) or [])
            if getattr(step, "type", None) == "function_call"
        ]

        if not tool_calls:
            return AgentResult(
                status="completed",
                output=last_output,
                steps_taken=steps_taken,
                tokens_used=tokens_used,
            )

        # Execute tools
        tool_results = []
        for call in tool_calls:
            tool_fn = TOOL_REGISTRY[call.name]
            result = tool_fn(**(call.arguments or {}))
            tool_results.append(
                {
                    "type": "function_result",
                    "name": call.name,
                    "call_id": getattr(call, "id", None),
                    "result": [
                        {
                            "type": "text",
                            "text": json.dumps({"ok": True, "result": result}),
                        }
                    ],
                }
            )

        steps_taken += 1

        if time.time() - start_time >= limits.deadline_seconds:
            return AgentResult(
                status="deadline_exceeded",
                output=last_output,
                steps_taken=steps_taken,
                tokens_used=tokens_used,
            )

        interaction = client.interactions.create(
            model=model,
            previous_interaction_id=interaction.id,
            input=tool_results,
        )


def _mock_function_call_step(tool_name: str = "get_time") -> MagicMock:
    step = MagicMock()
    step.type = "function_call"
    step.name = tool_name
    step.arguments = {}
    return step


@pytest.mark.asyncio
async def test_step_cap_limit():
    client = MagicMock()
    # Mock interaction that always returns a tool call to simulate a loop
    mock_interaction = MagicMock(
        steps=[_mock_function_call_step()],
        usage=MagicMock(total_tokens=10),
        output_text="In progress...",
    )
    client.interactions.create.return_value = mock_interaction

    limits = AgentLimits(max_steps=2, token_budget=1000, deadline_seconds=10.0)
    result = await run_agent(goal="Loop task", client=client, limits=limits)

    assert result.status == "max_steps_exceeded"
    assert result.steps_taken == 2
    assert result.output == "In progress..."


@pytest.mark.asyncio
async def test_token_budget_limit():
    client = MagicMock()
    # Mock interaction consuming 600 tokens
    mock_interaction = MagicMock(
        steps=[_mock_function_call_step()],
        usage=MagicMock(total_tokens=600),
        output_text="High token response",
    )
    client.interactions.create.return_value = mock_interaction

    limits = AgentLimits(max_steps=10, token_budget=500, deadline_seconds=10.0)
    result = await run_agent(goal="Expensive task", client=client, limits=limits)

    assert result.status == "token_budget_exceeded"
    assert result.tokens_used == 600
    assert result.output is None or result.output == "High token response"


@pytest.mark.asyncio
async def test_wall_clock_deadline_limit():
    client = MagicMock()

    # Simulate slow API call or tool execution
    def slow_interaction(*args, **kwargs):
        time.sleep(0.05)
        return MagicMock(
            steps=[_mock_function_call_step()],
            usage=MagicMock(total_tokens=5),
            output_text="Timed out work",
        )

    client.interactions.create.side_effect = slow_interaction

    limits = AgentLimits(max_steps=10, token_budget=1000, deadline_seconds=0.02)
    result = await run_agent(goal="Slow task", client=client, limits=limits)

    assert result.status == "deadline_exceeded"
