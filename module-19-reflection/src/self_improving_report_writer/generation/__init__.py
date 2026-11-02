"""Generation components for report synthesis."""

from .prompts import build_report_generation_prompt, format_source_contexts
from .report_generator import ReportGenerator

__all__ = [
    "ReportGenerator",
    "build_report_generation_prompt",
    "format_source_contexts",
]
