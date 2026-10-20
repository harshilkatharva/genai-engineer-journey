from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from ..core.settings import get_settings
from ..models.llm_model.llm_response_model import ToolCall
from ..observability import ObservabilityEvent, get_logger, safe_log
from .base import BaseTool
from .guards import ApprovalStore
from .models import (
    ApprovalRequest,
    RiskLevel,
    ToolExecutionStatus,
    ToolResult,
)
from .registry import ToolRegistry

_logger = get_logger("genai_core.tools.executor")


class ToolExecutor:
    """Executes tool calls with validation, timeouts, concurrency limits, and approval gating.

    Process-level Security & Isolation Notice:
    ToolExecutor provides in-process schema validation, timeout enforcement via asyncio,
    concurrency bounds via asyncio.Semaphore, output size truncation, and authorization gating.
    It executes inside the current Python process environment. It DOES NOT provide operating-system-level
    sandboxing, kernel isolation, memory virtualization, or container jails. Untrusted code or
    arbitrary bash/eval execution must be sandboxed at the OS or container infrastructure level.
    """

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        default_timeout_s: float | None = None,
        max_output_chars: int | None = None,
        max_concurrency: int | None = None,
        require_approval_for: set[RiskLevel] | None = None,
        approval_checker: (Callable[[ApprovalRequest], bool | Awaitable[bool]] | None) = None,
        approval_store: ApprovalStore | None = None,
    ) -> None:
        settings = get_settings()
        self.registry = registry or ToolRegistry()
        self.default_timeout_s = (
            default_timeout_s
            if default_timeout_s is not None
            else settings.tool_default_timeout_seconds
        )
        self.max_output_chars = (
            max_output_chars if max_output_chars is not None else settings.tool_max_output_chars
        )
        self.max_concurrency = (
            max_concurrency if max_concurrency is not None else settings.tool_max_concurrency
        )
        self._semaphore = asyncio.Semaphore(self.max_concurrency)

        self.require_approval_for = (
            require_approval_for if require_approval_for is not None else {RiskLevel.IRREVERSIBLE}
        )
        self.approval_checker = approval_checker
        self.approval_store = approval_store

    def register(self, tool: BaseTool | Callable[..., Any]) -> BaseTool:
        """Convenience method to register a tool into the underlying registry."""
        return self.registry.register(tool)

    async def _check_approval(
        self,
        call: ToolCall,
        tool: BaseTool,
        args_dict: dict[str, Any],
    ) -> tuple[bool, ApprovalRequest | None]:
        """Check if tool requires external approval and evaluate authorization."""
        if tool.risk_level not in self.require_approval_for:
            return True, None

        request = ApprovalRequest(
            call_id=call.id,
            tool_name=call.name,
            arguments=args_dict,
            risk_level=tool.risk_level,
            reason=f"Tool '{call.name}' has risk level '{tool.risk_level.value}' requiring authorization",
        )

        if self.approval_store:
            await self.approval_store.save_request(request)

        safe_log(
            _logger,
            30,  # WARNING
            "Tool execution paused: approval required",
            event=ObservabilityEvent.TOOL_APPROVAL_REQUIRED,
            tool_name=call.name,
            call_id=call.id,
            risk_level=tool.risk_level.value,
        )

        if self.approval_checker is not None:
            res = self.approval_checker(request)
            approved = await res if inspect.isawaitable(res) else res
            return bool(approved), request

        return False, request

    def _truncate_output(self, content: Any) -> tuple[Any, bool]:
        """Truncate overly large tool outputs to protect the LLM context window."""
        if self.max_output_chars <= 0:
            return content, False

        text: str
        if isinstance(content, str):
            text = content
        elif isinstance(content, BaseModel):
            text = content.model_dump_json()
        elif isinstance(content, (dict, list)):
            try:
                text = json.dumps(content, default=str)
            except (TypeError, ValueError):
                text = str(content)
        else:
            text = str(content)

        if len(text) > self.max_output_chars:
            marker = f"\n... [TRUNCATED: output exceeded {self.max_output_chars} characters]"
            truncated = text[: self.max_output_chars] + marker
            return truncated, True

        return content, False

    async def execute(
        self,
        call: ToolCall,
        timeout_s: float | None = None,
        bypass_approval: bool = False,
    ) -> ToolResult:
        """Execute a single ToolCall with safety policies, timeouts, and error handling."""
        start_time = time.perf_counter()
        effective_timeout = timeout_s if timeout_s is not None else self.default_timeout_s

        safe_log(
            _logger,
            20,  # INFO
            "Tool execution started",
            event=ObservabilityEvent.TOOL_EXECUTION_START,
            tool_name=call.name,
            call_id=call.id,
        )

        tool = self.registry.get(call.name)
        if tool is None:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            msg = f"Tool '{call.name}' is not registered."
            safe_log(
                _logger,
                40,  # ERROR
                msg,
                event=ObservabilityEvent.TOOL_EXECUTION_FAILED,
                tool_name=call.name,
                call_id=call.id,
                latency_ms=latency_ms,
            )
            return ToolResult(
                call_id=call.id,
                tool_name=call.name,
                status=ToolExecutionStatus.EXECUTION_ERROR,
                error_message=msg,
                latency_ms=latency_ms,
            )

        # 1. Schema Validation (Separated from Execution)
        spec = tool.to_spec()
        parsed_args, val_error = spec.parse_arguments(call.arguments)
        if val_error is not None:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            safe_log(
                _logger,
                30,  # WARNING
                f"Tool argument validation failed: {val_error.message}",
                event=ObservabilityEvent.TOOL_EXECUTION_FAILED,
                tool_name=call.name,
                call_id=call.id,
                validation_error=val_error.code,
                latency_ms=latency_ms,
            )
            return ToolResult(
                call_id=call.id,
                tool_name=call.name,
                status=ToolExecutionStatus.VALIDATION_ERROR,
                error_message=f"Validation failed: {val_error.message} (details: {val_error.details})",
                latency_ms=latency_ms,
            )

        args_dict = (
            parsed_args.model_dump() if isinstance(parsed_args, BaseModel) else (parsed_args or {})
        )

        # 2. Risk and Authorization Policy Checks
        if not bypass_approval:
            approved, approval_req = await self._check_approval(call, tool, args_dict)
            if not approved:
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                return ToolResult(
                    call_id=call.id,
                    tool_name=call.name,
                    status=ToolExecutionStatus.APPROVAL_REQUIRED,
                    content=approval_req.model_dump() if approval_req else None,
                    error_message=f"Execution of '{call.name}' paused: requires approval",
                    latency_ms=latency_ms,
                )

        # 3. Guarded Execution with Concurrency and Timeout
        async with self._semaphore:
            try:
                async with asyncio.timeout(effective_timeout):
                    raw_result = await tool.run(**args_dict)

                content, truncated = self._truncate_output(raw_result)
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

                safe_log(
                    _logger,
                    20,  # INFO
                    "Tool execution succeeded",
                    event=ObservabilityEvent.TOOL_EXECUTION_SUCCESS,
                    tool_name=call.name,
                    call_id=call.id,
                    latency_ms=latency_ms,
                    truncated=truncated,
                )
                return ToolResult(
                    call_id=call.id,
                    tool_name=call.name,
                    status=ToolExecutionStatus.SUCCESS,
                    content=content,
                    latency_ms=latency_ms,
                    truncated=truncated,
                )

            except TimeoutError:
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                msg = f"Tool '{call.name}' timed out after {effective_timeout}s."
                safe_log(
                    _logger,
                    40,  # ERROR
                    msg,
                    event=ObservabilityEvent.TOOL_EXECUTION_FAILED,
                    tool_name=call.name,
                    call_id=call.id,
                    latency_ms=latency_ms,
                )
                return ToolResult(
                    call_id=call.id,
                    tool_name=call.name,
                    status=ToolExecutionStatus.TIMEOUT,
                    error_message=msg,
                    latency_ms=latency_ms,
                )

            except Exception as exc:  # noqa: BLE001
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                msg = f"Tool execution failed: {type(exc).__name__}: {exc}"
                safe_log(
                    _logger,
                    40,  # ERROR
                    msg,
                    event=ObservabilityEvent.TOOL_EXECUTION_FAILED,
                    tool_name=call.name,
                    call_id=call.id,
                    error_type=type(exc).__name__,
                    latency_ms=latency_ms,
                )
                return ToolResult(
                    call_id=call.id,
                    tool_name=call.name,
                    status=ToolExecutionStatus.EXECUTION_ERROR,
                    error_message=msg,
                    latency_ms=latency_ms,
                )

    async def execute_many(
        self,
        calls: list[ToolCall],
        timeout_s: float | None = None,
    ) -> list[ToolResult]:
        """Execute multiple tool calls concurrently within configured concurrency bounds."""
        if not calls:
            return []
        tasks = [self.execute(call, timeout_s=timeout_s) for call in calls]
        return await asyncio.gather(*tasks)
