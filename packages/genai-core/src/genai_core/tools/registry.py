from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from typing import Any

from ..exceptions.tool_exceptions import ToolNotFoundError
from ..models.tools_model.tool_spec import ToolSpec
from .base import BaseTool, FunctionTool
from .models import RiskLevel

_RISK_HIERARCHY: dict[RiskLevel, int] = {
    RiskLevel.READ_ONLY: 0,
    RiskLevel.REVERSIBLE_WRITE: 1,
    RiskLevel.IRREVERSIBLE: 2,
}


class ToolRegistry:
    """Thread-safe registry for managing, cataloging, and filtering tools."""

    def __init__(self, tools: list[BaseTool] | None = None) -> None:
        self._lock = threading.RLock()
        self._tools: dict[str, BaseTool] = {}
        if tools:
            for t in tools:
                self.register(t)

    def register(
        self,
        tool: BaseTool | Callable[..., Any],
        override: bool = False,
    ) -> BaseTool:
        """Register a tool instance or callable function into the registry."""
        tool_instance: BaseTool
        if isinstance(tool, BaseTool):
            tool_instance = tool
        elif callable(tool):
            tool_instance = FunctionTool(tool)
        else:
            raise TypeError(f"Expected BaseTool or Callable, got {type(tool).__name__}")

        with self._lock:
            name = tool_instance.name
            if name in self._tools and not override:
                raise ValueError(
                    f"Tool '{name}' is already registered. Set override=True to replace it."
                )
            self._tools[name] = tool_instance
            return tool_instance

    def unregister(self, name: str) -> BaseTool | None:
        """Remove a tool by name if registered."""
        with self._lock:
            return self._tools.pop(name, None)

    def get(self, name: str) -> BaseTool | None:
        """Retrieve a registered tool by name, returning None if not found."""
        with self._lock:
            return self._tools.get(name)

    def get_or_fail(self, name: str) -> BaseTool:
        """Retrieve a registered tool by name or raise ToolNotFoundError."""
        with self._lock:
            tool = self._tools.get(name)
            if tool is None:
                raise ToolNotFoundError(f"Tool '{name}' is not registered.")
            return tool

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        with self._lock:
            return name in self._tools

    def list_tools(self) -> list[BaseTool]:
        """Return a snapshot list of all registered tools."""
        with self._lock:
            return list(self._tools.values())

    def get_specs(
        self,
        tags: set[str] | list[str] | None = None,
        max_risk: RiskLevel | None = None,
        exclude: set[str] | list[str] | None = None,
    ) -> list[ToolSpec]:
        """Generate ToolSpec models for LLM consumption with optional filtering."""
        with self._lock:
            tag_set = set(tags) if tags is not None else None
            exclude_set = set(exclude) if exclude is not None else set()
            max_risk_level = _RISK_HIERARCHY[max_risk] if max_risk is not None else None

            specs: list[ToolSpec] = []
            for tool in self._tools.values():
                if tool.name in exclude_set:
                    continue

                if max_risk_level is not None and _RISK_HIERARCHY[tool.risk_level] > max_risk_level:
                    continue

                if tag_set is not None and not (tool.tags & tag_set):
                    continue

                specs.append(tool.to_spec())

            return specs

    def clear(self) -> None:
        """Remove all registered tools."""
        with self._lock:
            self._tools.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return self.has(name)

    def __iter__(self) -> Iterator[BaseTool]:
        with self._lock:
            return iter(list(self._tools.values()))
