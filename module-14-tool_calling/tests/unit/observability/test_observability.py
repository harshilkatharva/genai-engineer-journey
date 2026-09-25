import logging
from uuid import uuid4

from customer_support_agent.observability.context import (
    get_request_id,
    reset_request_id,
    set_request_id,
)
from customer_support_agent.observability.events import EventName
from customer_support_agent.observability.logger import JsonFormatter
from customer_support_agent.observability.tool_logger import ToolInvocationLogger


def test_request_context_round_trip_and_events():
    request_id = uuid4()
    token = set_request_id(request_id)
    try:
        assert get_request_id() == request_id
    finally:
        reset_request_id(token)
    assert get_request_id() is None
    assert EventName.TOOL_FAILED.value == "tool_failed"


def test_json_formatter_and_tool_logger_emit_structured_records(monkeypatch):
    record = logging.LogRecord("test", logging.INFO, "", 1, "hello", (), None)
    record.event = EventName.TOOL_INVOKED
    assert '"message": "hello"' in JsonFormatter().format(record)
    calls = []
    monkeypatch.setattr(
        "customer_support_agent.observability.tool_logger.logger.info",
        lambda *a, **kw: calls.append(kw),
    )
    monkeypatch.setattr(
        "customer_support_agent.observability.tool_logger.logger.error",
        lambda *a, **kw: calls.append(kw),
    )
    logger = ToolInvocationLogger()
    started = logger.start("lookup_order", {}, 1)
    logger.complete("lookup_order", started, 1)
    logger.fail("lookup_order", started, 1, ValueError("bad"))
    assert len(calls) == 3
    assert calls[-1]["error_type"] == "ValueError"
