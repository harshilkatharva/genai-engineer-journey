"""Prompt construction utilities with prompt-injection defense boundaries."""

from __future__ import annotations

from collections.abc import Sequence

from self_improving_report_writer.models.report import SourceReference
from self_improving_report_writer.prompts.manager import get_prompt_manager

_prompt_mgr = get_prompt_manager()
SYSTEM_PROMPT = _prompt_mgr.get_system_prompt("generation", "system")


def format_source_contexts(sources: Sequence[SourceReference]) -> str:
    """Format source chunks inside strict XML-like data boundaries."""
    return get_prompt_manager().format_source_contexts(sources)


def build_report_generation_prompt(
    topic: str,
    sources: Sequence[SourceReference],
    guidelines: list[str] | None = None,
) -> str:
    """Build complete structured report generation prompt."""
    prompt_mgr = get_prompt_manager()
    user_prompt = prompt_mgr.render_report_generation_prompt(
        topic=topic,
        sources=sources,
        guidelines=guidelines,
    )
    return f"{SYSTEM_PROMPT}\n\n{user_prompt}"
