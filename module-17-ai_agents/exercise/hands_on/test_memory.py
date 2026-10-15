"""Task 6 – memory system tests (long-term recall + episodic recording)."""

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

# ── Inline implementation (mirrors notebook) ──────────────────────────────────


@dataclass
class MemoryStore:
    long_term: dict[str, str] = field(default_factory=dict)
    episodes: list[dict] = field(default_factory=list)

    def remember(self, key: str, value: str) -> None:
        self.long_term[key] = value

    def recall_all(self) -> dict[str, str]:
        return dict(self.long_term)

    def record(self, tool: str, args: dict[str, Any], result: Any) -> None:
        self.episodes.append({"tool": tool, "args": args, "result": result})

    def history(self) -> list[dict[str, Any]]:
        return list(self.episodes)


TOOL_REGISTRY: dict[str, Any] = {
    "get_time": lambda: "2026-10-15T12:00:00",
    "calculator": lambda a, b, operation: a + b,
}

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "name": "get_time",
        "description": "Get time.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "calculator",
        "description": "Do arithmetic.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
                "operation": {"type": "string"},
            },
            "required": ["a", "b", "operation"],
        },
    },
]


async def run_agent_with_memory(
    goal: str,
    client: Any,
    model: str,
    memory: MemoryStore,
) -> str:
    facts = memory.recall_all()
    context = "\n".join(f"- {k}: {v}" for k, v in facts.items())

    interaction = client.interactions.create(
        model=model,
        input=goal,
        tools=TOOL_DEFINITIONS,
        system_instruction=f"Known context:\n{context}",
    )

    for _ in range(6):
        calls = [s for s in (interaction.steps or []) if s.type == "function_call"]
        if not calls:
            return interaction.output_text or ""

        results = []
        for call in calls:
            args = call.arguments or {}
            try:
                outcome = TOOL_REGISTRY[call.name](**args)
                payload: dict[str, Any] = {"ok": True, "result": outcome}
            except Exception as exc:  # noqa: BLE001
                outcome = str(exc)
                payload = {"ok": False, "error": outcome}

            memory.record(call.name, args, outcome)
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
            tools=TOOL_DEFINITIONS,
        )

    return ""


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_call(name: str, **kwargs: Any) -> MagicMock:
    call = MagicMock()
    call.type = "function_call"
    call.name = name
    call.arguments = kwargs
    call.id = f"id_{name}"
    return call


def _interaction(calls: list[MagicMock], output: str = "done") -> MagicMock:
    m = MagicMock()
    m.steps = calls
    m.output_text = output
    return m


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_remember_and_recall() -> None:
    mem = MemoryStore()
    mem.remember("user_tz", "Asia/Kolkata")
    mem.remember("unit", "metric")
    assert mem.recall_all() == {"user_tz": "Asia/Kolkata", "unit": "metric"}


def test_recall_returns_copy() -> None:
    mem = MemoryStore()
    mem.remember("key", "val")
    snapshot = mem.recall_all()
    snapshot["key"] = "mutated"
    assert mem.recall_all()["key"] == "val"  # original unchanged


@pytest.mark.asyncio
async def test_long_term_memory_injected_into_system_prompt() -> None:
    """Verify long-term facts reach the model via system_instruction."""
    client = MagicMock()
    client.interactions.create.return_value = _interaction([], output="done")

    mem = MemoryStore()
    mem.remember("user_tz", "Asia/Kolkata")

    await run_agent_with_memory("What time is it?", client, "test-model", mem)

    first_call_kwargs = client.interactions.create.call_args_list[0].kwargs
    assert "Asia/Kolkata" in first_call_kwargs["system_instruction"]


@pytest.mark.asyncio
async def test_tool_call_recorded_to_episodic_memory() -> None:
    """Every tool executed must appear in episodic history."""
    client = MagicMock()
    client.interactions.create.side_effect = [
        _interaction([_make_call("get_time")]),
        _interaction([], output="The time is noon."),
    ]

    mem = MemoryStore()
    await run_agent_with_memory("What time is it?", client, "test-model", mem)

    assert len(mem.history()) == 1
    assert mem.history()[0]["tool"] == "get_time"
    assert mem.history()[0]["result"] == "2026-10-15T12:00:00"


@pytest.mark.asyncio
async def test_multiple_tools_all_recorded() -> None:
    """Multiple tool calls across steps all end up in episodic memory."""
    client = MagicMock()
    client.interactions.create.side_effect = [
        _interaction([_make_call("get_time"), _make_call("calculator", a=3, b=4, operation="add")]),
        _interaction([], output="Done."),
    ]

    mem = MemoryStore()
    await run_agent_with_memory("Time and 3+4?", client, "test-model", mem)

    tools_run = [ep["tool"] for ep in mem.history()]
    assert tools_run == ["get_time", "calculator"]


@pytest.mark.asyncio
async def test_failed_tool_still_recorded() -> None:
    """Even a tool that raises an error is captured in episodic memory."""
    client = MagicMock()
    bad_call = _make_call("calculator", a=1, b=0, operation="divide")
    client.interactions.create.side_effect = [
        _interaction([bad_call]),
        _interaction([], output="Cannot divide."),
    ]

    TOOL_REGISTRY["calculator"] = lambda a, b, operation: 1 / 0  # force error

    mem = MemoryStore()
    await run_agent_with_memory("Divide 1 by 0", client, "test-model", mem)

    assert len(mem.history()) == 1
    assert "division by zero" in str(mem.history()[0]["result"])

    # Restore original
    TOOL_REGISTRY["calculator"] = lambda a, b, operation: a + b
