"""Centralized prompt templates and manager loaded via Jinja2."""

from self_improving_report_writer.prompts.manager import (
    PromptTemplateManager,
    escape_xml_context,
    get_prompt_manager,
)

__all__ = [
    "PromptTemplateManager",
    "escape_xml_context",
    "get_prompt_manager",
]
