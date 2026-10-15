"""
LangGraph re-implementation of the same agent from Task 1.

Same tools (get_time, calculator), same goal, different framework.
Run with: python3 exercise/hands_on/agent_langgraph.py
"""

import os
import warnings
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.tools import tool as lc_tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

# Silence the LangGraph v1→v2 deprecation notice cleanly
warnings.filterwarnings("ignore", message=".*create_react_agent.*", category=DeprecationWarning)

load_dotenv()

# ── Tools (same logic as Task 1) ──────────────────────────────────────────────


@lc_tool
def get_time() -> str:
    """Get the current local date and time."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


@lc_tool
def calculator(a: float, b: float, operation: str) -> float:
    """Perform basic arithmetic. operation: add | subtract | multiply | divide."""
    ops: dict[str, float] = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b,
    }
    if operation not in ops:
        raise ValueError(f"Unknown operation: {operation}")
    return ops[operation]


# ── Build the graph ───────────────────────────────────────────────────────────


def build_agent() -> CompiledStateGraph:
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    return create_react_agent(llm, tools=[get_time, calculator])


def run(goal: str) -> str:
    graph = build_agent()
    result = graph.invoke({"messages": [("user", goal)]})
    # Last message is the final AI response
    last = result["messages"][-1]
    # content may be str or list[dict] depending on model/version
    if isinstance(last.content, str):
        return last.content
    # extract text parts from structured content
    return " ".join(
        part["text"] for part in last.content if isinstance(part, dict) and "text" in part
    )


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    goal = "Calculate the sum of 5 and 10, then get the current time. Report both."
    print("Goal:", goal)
    print("\nAnswer:", run(goal))
