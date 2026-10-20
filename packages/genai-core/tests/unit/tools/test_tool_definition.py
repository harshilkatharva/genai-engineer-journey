from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from genai_core.tools import BaseTool, FunctionTool, RiskLevel, tool


class AddArgs(BaseModel):
    x: int
    y: int = Field(default=10)


class CustomCalculator(BaseTool):
    name = "custom_calc"
    description = "A custom calculator tool"
    args_schema = AddArgs
    risk_level = RiskLevel.READ_ONLY

    async def run(self, **kwargs: int) -> int:
        return kwargs["x"] + kwargs["y"]


@pytest.mark.asyncio
async def test_base_tool_subclass():
    calc = CustomCalculator()
    assert calc.name == "custom_calc"
    assert calc.risk_level == RiskLevel.READ_ONLY
    assert len(calc.tags) == 0

    calc.tags = {"math", "calc"}
    assert "math" in calc.tags

    result = await calc.run(x=5, y=15)
    assert result == 20

    spec = calc.to_spec()
    assert spec.name == "custom_calc"
    assert "x" in spec.parameters["properties"]
    assert "y" in spec.parameters["properties"]


@pytest.mark.asyncio
async def test_tool_decorator_sync_function():
    @tool
    def multiply(a: int, b: int = 2) -> int:
        """Multiply two numbers."""
        return a * b

    assert isinstance(multiply, FunctionTool)
    assert multiply.name == "multiply"
    assert "Multiply two numbers." in multiply.description
    assert multiply.risk_level == RiskLevel.READ_ONLY

    # Run sync function asynchronously
    res = await multiply.run(a=6, b=7)
    assert res == 42


@pytest.mark.asyncio
async def test_tool_decorator_async_function():
    @tool(
        name="fetch_user",
        description="Fetch a user by ID",
        risk_level=RiskLevel.REVERSIBLE_WRITE,
        tags=["users", "api"],
    )
    async def fetch_user(user_id: str) -> dict[str, str]:
        return {"id": user_id, "name": f"User_{user_id}"}

    assert fetch_user.name == "fetch_user"
    assert fetch_user.description == "Fetch a user by ID"
    assert fetch_user.risk_level == RiskLevel.REVERSIBLE_WRITE
    assert fetch_user.tags == {"users", "api"}

    res = await fetch_user.run(user_id="u-123")
    assert res == {"id": "u-123", "name": "User_u-123"}


@pytest.mark.asyncio
async def test_tool_decorator_with_explicit_schema():
    class SearchArgs(BaseModel):
        query: str
        limit: int = 5

    @tool(args_schema=SearchArgs)
    def search_docs(query: str, limit: int = 5) -> list[str]:
        return [f"result_{i}" for i in range(limit)]

    assert search_docs.args_schema == SearchArgs
    spec = search_docs.to_spec()
    parsed, err = spec.parse_arguments({"query": "AI", "limit": 2})
    assert err is None
    assert parsed is not None
    assert parsed.query == "AI"  # type: ignore[attr-defined]
    assert parsed.limit == 2  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_parameterless_tool():
    @tool
    def ping() -> str:
        """Ping check."""
        return "pong"

    spec = ping.to_spec()
    _parsed, err = spec.parse_arguments({})
    assert err is None
    res = await ping.run()
    assert res == "pong"
