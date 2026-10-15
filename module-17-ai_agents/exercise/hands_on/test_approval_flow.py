"""Task 5 – pause-for-approval flow tests (approve & reject paths)."""

import json
from enum import Enum
from typing import Any
from unittest.mock import MagicMock

import pytest

# ── Inline implementation (mirrors the notebook) ──────────────────────────────


class Risk(Enum):
    LOW = "low"
    HIGH = "high"


TOOLS_WITH_RISK: dict[str, tuple[Any, Risk]] = {
    "get_time": (lambda: "2026-10-14T18:00:00", Risk.LOW),
    "delete_file": (lambda path: f"Deleted {path}", Risk.HIGH),
    "send_email": (lambda to, body: f"Sent to {to}", Risk.HIGH),
}

TOOL_DEFS = [
    {
        "type": "function",
        "name": "get_time",
        "description": "Get time.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "delete_file",
        "description": "Delete a file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "type": "function",
        "name": "send_email",
        "description": "Send an email.",
        "parameters": {
            "type": "object",
            "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
            "required": ["to", "body"],
        },
    },
]


async def run_agent_with_approval(
    goal: str,
    client: Any,
    model: str = "test-model",
    approve_fn: Any = lambda name, args: True,
) -> dict[str, Any]:
    interaction = client.interactions.create(model=model, input=goal, tools=TOOL_DEFS)

    for _ in range(6):
        calls = [s for s in (interaction.steps or []) if s.type == "function_call"]
        if not calls:
            return {"status": "completed", "output": interaction.output_text}

        results = []
        for call in calls:
            fn, risk = TOOLS_WITH_RISK[call.name]
            args = call.arguments or {}

            if risk == Risk.HIGH and not approve_fn(call.name, args):
                payload: dict[str, Any] = {"ok": False, "error": f"`{call.name}` rejected by user."}
            else:
                try:
                    payload = {"ok": True, "result": fn(**args)}
                except Exception as exc:  # noqa
                    payload = {"ok": False, "error": str(exc)}

            results.append(
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
            input=results,
            tools=TOOL_DEFS,
        )

    return {"status": "max_steps_exceeded"}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_call(name: str, **kwargs: Any) -> MagicMock:
    call = MagicMock()
    call.type = "function_call"
    call.name = name
    call.arguments = kwargs
    call.id = f"call_{name}"
    return call


def _interaction(calls: list[MagicMock], output: str = "") -> MagicMock:
    inter = MagicMock()
    inter.steps = calls
    inter.output_text = output
    return inter


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_low_risk_tool_runs_without_approval() -> None:
    """LOW-risk tools execute immediately, no approval needed."""
    client = MagicMock()
    client.interactions.create.side_effect = [
        _interaction([_make_call("get_time")]),  # step 0: calls get_time
        _interaction([], output="The time is …"),  # step 1: final answer
    ]

    result = await run_agent_with_approval(
        "What time is it?",
        client,
        approve_fn=lambda name, args: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_high_risk_tool_runs_when_approved() -> None:
    """HIGH-risk tool executes when approve_fn returns True."""
    client = MagicMock()
    client.interactions.create.side_effect = [
        _interaction([_make_call("delete_file", path="/tmp/old.txt")]),
        _interaction([], output="File deleted."),
    ]

    approved_tools: list[str] = []

    def approve(name: str, args: dict[str, Any]) -> bool:
        approved_tools.append(name)
        return True

    result = await run_agent_with_approval("Delete /tmp/old.txt", client, approve_fn=approve)

    assert result["status"] == "completed"
    assert "delete_file" in approved_tools

    # Check the tool result sent back was ok=True
    tool_result_call = client.interactions.create.call_args_list[1]
    payload = json.loads(tool_result_call.kwargs["input"][0]["result"][0]["text"])
    assert payload["ok"] is True
    assert "Deleted" in payload["result"]


@pytest.mark.asyncio
async def test_high_risk_tool_blocked_when_rejected() -> None:
    """HIGH-risk tool is blocked and an error is returned to the model when rejected."""
    client = MagicMock()
    client.interactions.create.side_effect = [
        _interaction([_make_call("send_email", to="bob@example.com", body="hi")]),
        _interaction([], output="I couldn't send the email – it was rejected."),
    ]

    result = await run_agent_with_approval(
        "Send email to bob@example.com",
        client,
        approve_fn=lambda name, args: False,  # always reject
    )

    assert result["status"] == "completed"

    # Check the tool result sent back was ok=False with rejection error
    tool_result_call = client.interactions.create.call_args_list[1]
    payload = json.loads(tool_result_call.kwargs["input"][0]["result"][0]["text"])
    assert payload["ok"] is False
    assert "rejected" in payload["error"]
