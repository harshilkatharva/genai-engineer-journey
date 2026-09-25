from enum import StrEnum


class EventName(StrEnum):
    LLM_COMPLETED = "llm_completed"
    LLM_FAILED = "llm_failed"
    TOOL_INVOKED = "tool_invoked"
    TOOL_FAILED = "tool_failed"
    DB_QUERY = "db_query"
