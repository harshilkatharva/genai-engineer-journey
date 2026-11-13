"""Generation components for report synthesis."""

from self_improving_report_writer.generation.prompts import (
    build_report_generation_prompt,
    format_source_contexts,
)
from self_improving_report_writer.generation.report_generator import ReportGenerator

__all__ = [
    "ReportGenerator",
    "build_report_generation_prompt",
    "format_source_contexts",
]
