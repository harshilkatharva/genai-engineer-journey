from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from ..models.tools_model.tool_spec import ToolSpec
from .models import RiskLevel


class BaseTool(ABC):
    """Abstract base class for all tools in genai-core."""

    name: str
    description: str
    args_schema: type[BaseModel]
    risk_level: RiskLevel = RiskLevel.READ_ONLY
    _tags: set[str] | None = None

    @property
    def tags(self) -> set[str]:
        """Tags associated with the tool for discovery and categorization."""
        if self._tags is None:
            self._tags = set()
        return self._tags

    @tags.setter
    def tags(self, value: set[str]) -> None:
        self._tags = value

    def to_spec(self) -> ToolSpec:
        """Export tool metadata and schema as a reusable ToolSpec."""
        return ToolSpec(
            name=self.name,
            description=self.description,
            argument_model=self.args_schema,
            handler=self.run,
        )

    @abstractmethod
    async def run(self, **kwargs: Any) -> Any:
        """Execute the tool with validated arguments."""
        ...


def _infer_args_schema_from_callable(func: Callable[..., Any], model_name: str) -> type[BaseModel]:
    """Dynamically build a Pydantic model from a function's parameter signature."""
    sig = inspect.signature(func)
    fields: dict[str, Any] = {}

    for param in sig.parameters.values():
        if param.name in ("self", "cls"):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        annotation = param.annotation if param.annotation != inspect.Parameter.empty else Any
        if param.default != inspect.Parameter.empty:
            fields[param.name] = (annotation, Field(default=param.default))
        else:
            fields[param.name] = (annotation, ...)

    if not fields:
        # Parameterless tool
        class EmptySchema(BaseModel):
            model_config = ConfigDict(extra="ignore")

        EmptySchema.__name__ = model_name
        return EmptySchema

    return create_model(model_name, **fields)  # type: ignore[no-any-return,call-overload]


class FunctionTool(BaseTool):
    """Tool implementation wrapping a standalone synchronous or asynchronous function."""

    def __init__(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
        args_schema: type[BaseModel] | None = None,
        risk_level: RiskLevel = RiskLevel.READ_ONLY,
        tags: set[str] | list[str] | None = None,
    ) -> None:
        self._func = func
        self.name = name or func.__name__
        self.description = (
            description or inspect.getdoc(func) or f"Tool executing function {self.name}"
        ).strip()
        schema_name = f"{''.join(word.capitalize() for word in self.name.split('_'))}Args"
        self.args_schema = args_schema or _infer_args_schema_from_callable(func, schema_name)
        self.risk_level = risk_level
        self.tags = set(tags or [])

    async def run(self, **kwargs: Any) -> Any:
        """Invoke wrapped callable, running sync functions in thread pool."""
        if asyncio.iscoroutinefunction(self._func):
            return await self._func(**kwargs)
        return await asyncio.to_thread(self._func, **kwargs)


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    args_schema: type[BaseModel] | None = None,
    risk_level: RiskLevel = RiskLevel.READ_ONLY,
    tags: set[str] | list[str] | None = None,
) -> Any:
    """Decorator to declare a tool from a sync or async function.

    Can be used with or without arguments:
        @tool
        def calculate(a: int, b: int) -> int: ...

        @tool(name="query_db", risk_level=RiskLevel.IRREVERSIBLE)
        async def query_db(query: str) -> list[dict]: ...
    """

    def decorator(fn: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(
            func=fn,
            name=name,
            description=description,
            args_schema=args_schema,
            risk_level=risk_level,
            tags=tags,
        )

    if func is not None:
        return decorator(func)
    return decorator
