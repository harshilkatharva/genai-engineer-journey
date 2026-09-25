from __future__ import annotations

from time import perf_counter
from typing import Any

from .events import EventName
from .logger import logger


class ToolInvocationLogger:
    def start(self, tool_name: str, arguments: dict[str, Any], iteration: int) -> float:
        started = perf_counter()
        logger.info(
            "Tool invocation started",
            event=EventName.TOOL_INVOKED,
            component="tool_calling",
            tool_name=tool_name,
            arguments=arguments,
            iteration=iteration,
            success=False,
        )
        return started

    def complete(self, tool_name: str, started: float, iteration: int) -> None:
        logger.info(
            "Tool invocation completed",
            event=EventName.TOOL_INVOKED,
            component="tool_calling",
            tool_name=tool_name,
            duration_ms=(perf_counter() - started) * 1000,
            iteration=iteration,
            success=True,
        )

    def fail(self, tool_name: str, started: float, iteration: int, error: Exception) -> None:
        logger.error(
            "Tool invocation failed",
            event=EventName.TOOL_FAILED,
            component="tool_calling",
            tool_name=tool_name,
            duration_ms=(perf_counter() - started) * 1000,
            iteration=iteration,
            success=False,
            error_type=type(error).__name__,
            error_message=str(error),
        )
