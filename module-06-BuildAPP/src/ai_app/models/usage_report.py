from dataclasses import dataclass


@dataclass
class UsageBreakdown:
    name: str
    message_count: int
    input_tokens: int
    output_tokens: int
    estimated_cost: float


@dataclass
class UsageReport:
    total_input_tokens: int
    total_output_tokens: int
    total_estimated_cost: float
    total_messages: int
    breakdown: list[UsageBreakdown]
