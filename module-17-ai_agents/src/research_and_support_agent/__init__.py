"""Module 17: Bounded Research and Support Agent."""

from .agent import AgentRunResult, BoundedAgent
from .core.config import AgentConfig, get_agent_config
from .hitl.authorization import AuthorizationValidator
from .hitl.postgres_store import PostgresApprovalStore
from .memory.manager import AgentMemoryManager
from .models import CustomerAccount, ReversalAuditRecord, SupportTicket, TicketStatus
from .telemetry.pricing import PricingCalculator
from .telemetry.transcript import AgentTranscript, TranscriptRecorder, TranscriptStep
from .tools.registry import create_agent_tool_registry
from .tools.sanitizer import ToolOutputSanitizer, sanitize_tool_output

__all__ = [
    "AgentConfig",
    "AgentMemoryManager",
    "AgentRunResult",
    "AgentTranscript",
    "AuthorizationValidator",
    "BoundedAgent",
    "CustomerAccount",
    "PostgresApprovalStore",
    "PricingCalculator",
    "ReversalAuditRecord",
    "SupportTicket",
    "TicketStatus",
    "ToolOutputSanitizer",
    "TranscriptRecorder",
    "TranscriptStep",
    "create_agent_tool_registry",
    "get_agent_config",
    "sanitize_tool_output",
]
