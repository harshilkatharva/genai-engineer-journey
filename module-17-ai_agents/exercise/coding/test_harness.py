"""Task 9: Offline Test Harness with Scripted Fake LLM.

Allows unit-testing agent loops, tool dispatch, and state handling
with a fixed sequence of tool calls and no network connection.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass
class ToolCallStep:
    """Represents a function call step produced by the LLM."""

    name: str
    arguments: dict[str, Any]
    id: str = "call_default"
    type: str = "function_call"


@dataclass
class FakeInteraction:
    """Represents an interaction turn returned by the LLM."""

    id: str = "interaction_default"
    steps: list[ToolCallStep] = field(default_factory=list)
    output_text: str = ""


class FakeLLMClient:
    """Scripted fake LLM client simulating google.genai.Client interactions offline."""

    def __init__(self, scripted_responses: list[FakeInteraction]) -> None:
        self.scripted_responses = list(scripted_responses)
        self.call_history: list[dict[str, Any]] = []
        self._current_index = 0

        # Sub-API mirroring client.interactions.create
        self.interactions = self

    def create(
        self,
        model: str,
        input: Any = None,
        tools: list[dict[str, Any]] | None = None,
        previous_interaction_id: str | None = None,
        **kwargs: Any,
    ) -> FakeInteraction:
        """Record the call and return the next scripted FakeInteraction."""
        self.call_history.append(
            {
                "model": model,
                "input": input,
                "tools": tools,
                "previous_interaction_id": previous_interaction_id,
                **kwargs,
            }
        )

        if self._current_index >= len(self.scripted_responses):
            raise IndexError(
                f"FakeLLMClient exhausted scripted responses: requested call "
                f"{self._current_index + 1}, but only had {len(self.scripted_responses)} configured."
            )

        response = self.scripted_responses[self._current_index]
        self._current_index += 1
        return response


# ── Core Agent Loop Under Test ───────────────────────────────────────────────


async def run_agent(
    goal: str,
    client: Any,
    tool_registry: dict[str, Callable[..., Any]],
    model: str = "fake-model",
    max_steps: int = 5,
) -> str:
    """Execute the observe -> LLM -> execute tools -> append loop."""
    interaction = client.interactions.create(model=model, input=goal)

    for _ in range(max_steps):
        tool_calls = [step for step in (interaction.steps or []) if step.type == "function_call"]

        # Exit condition: model produced final answer without requesting tools
        if not tool_calls:
            return interaction.output_text or ""

        tool_results = []
        for call in tool_calls:
            tool_fn = tool_registry[call.name]
            try:
                res = tool_fn(**(call.arguments or {}))
                payload = {"ok": True, "result": res}
            except Exception as exc:  # noqa: BLE001
                payload = {"ok": False, "error": str(exc)}

            tool_results.append(
                {
                    "type": "function_result",
                    "name": call.name,
                    "call_id": call.id,
                    "result": [{"type": "text", "text": json.dumps(payload)}],
                }
            )

        interaction = client.interactions.create(
            model=model,
            previous_interaction_id=interaction.id,
            input=tool_results,
        )

    raise RuntimeError("Max steps reached without reaching a final answer.")


# ── Unit Tests with No Network ───────────────────────────────────────────────


@pytest.fixture
def tools() -> dict[str, Callable[..., Any]]:
    return {
        "get_stock_price": lambda ticker: 235.50 if ticker.upper() == "AAPL" else 150.0,
        "convert_currency": lambda amount, from_currency, to_currency: round(
            (amount * 1.0) / 1.08, 2
        ),
    }


@pytest.mark.asyncio
async def test_scripted_single_tool_execution(tools: dict[str, Callable[..., Any]]) -> None:
    """Test a scripted LLM requesting a single tool and then outputting the answer."""
    script = [
        # Turn 0: LLM requests tool call
        FakeInteraction(
            id="turn-0",
            steps=[ToolCallStep(name="get_stock_price", arguments={"ticker": "AAPL"}, id="c-1")],
        ),
        # Turn 1: LLM returns final answer
        FakeInteraction(
            id="turn-1",
            steps=[],
            output_text="The stock price of AAPL is $235.50 USD.",
        ),
    ]

    client = FakeLLMClient(script)
    answer = await run_agent(
        goal="What is AAPL stock price?",
        client=client,
        tool_registry=tools,
    )

    # 1. Assert correct final answer
    assert answer == "The stock price of AAPL is $235.50 USD."

    # 2. Assert interaction calls
    assert len(client.call_history) == 2
    assert client.call_history[0]["input"] == "What is AAPL stock price?"

    # 3. Assert tool result passed into turn 1
    turn1_input = client.call_history[1]["input"]
    assert turn1_input[0]["name"] == "get_stock_price"
    payload = json.loads(turn1_input[0]["result"][0]["text"])
    assert payload["ok"] is True
    assert payload["result"] == 235.50


@pytest.mark.asyncio
async def test_scripted_multi_tool_chain(tools: dict[str, Callable[..., Any]]) -> None:
    """Test a scripted LLM chaining two sequential tool calls before final answer."""
    script = [
        # Turn 0: Call tool 1
        FakeInteraction(
            id="turn-0",
            steps=[ToolCallStep(name="get_stock_price", arguments={"ticker": "AAPL"}, id="c-1")],
        ),
        # Turn 1: Call tool 2 with output from tool 1
        FakeInteraction(
            id="turn-1",
            steps=[
                ToolCallStep(
                    name="convert_currency",
                    arguments={"amount": 235.50, "from_currency": "USD", "to_currency": "EUR"},
                    id="c-2",
                )
            ],
        ),
        # Turn 2: Final answer
        FakeInteraction(
            id="turn-2",
            steps=[],
            output_text="AAPL price is 218.06 EUR.",
        ),
    ]

    client = FakeLLMClient(script)
    answer = await run_agent(
        goal="Get AAPL price and convert to EUR",
        client=client,
        tool_registry=tools,
    )

    assert answer == "AAPL price is 218.06 EUR."
    assert len(client.call_history) == 3

    # Verify tool 2 result
    turn2_input = client.call_history[2]["input"]
    payload = json.loads(turn2_input[0]["result"][0]["text"])
    assert payload["ok"] is True
    assert payload["result"] == 218.06


@pytest.mark.asyncio
async def test_scripted_immediate_final_answer(tools: dict[str, Callable[..., Any]]) -> None:
    """Test scripted LLM returning a direct answer without tool calls."""
    script = [
        FakeInteraction(
            id="turn-0",
            steps=[],
            output_text="Direct answer with no tool needed.",
        )
    ]

    client = FakeLLMClient(script)
    answer = await run_agent(goal="Hello!", client=client, tool_registry=tools)

    assert answer == "Direct answer with no tool needed."
    assert len(client.call_history) == 1
