"""Prompts for deliverable synthesis and orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import Plan


def build_synthesis_prompt(
    goal: str,
    step_results: dict[str, str],
    plan: Plan,
    user_preferences: list[str] | None = None,
) -> str:
    """Build prompt for synthesizing final verified deliverable from completed plan steps."""
    lines: list[str] = [
        "You are an expert AI research assistant and operations synthesizer.",
        (
            "Your task is to synthesize a comprehensive, rigorous, and cohesive final deliverable "
            "that fulfills the user's goal based on verified step outputs."
        ),
        "",
        f"USER GOAL: {goal}",
    ]

    if user_preferences:
        lines.append("")
        lines.append("USER PREFERENCES & FORMATTING REQUIREMENTS:")
        for pref in user_preferences:
            lines.append(f"- {pref}")

    lines.append("")
    lines.append("COMPLETED & VERIFIED STEP DELIVERABLES:")
    for step in plan.steps:
        output = step_results.get(step.id, step.result_summary or "Completed.")
        lines.append(f"- Step [{step.id}] ({step.description}):")
        lines.append(f"  {output.strip()}")

    lines.extend(
        [
            "",
            "SYNTHESIS GUIDELINES:",
            "1. Seamlessly integrate the findings, calculations, actions, and facts from all completed steps.",
            "2. Address the original user goal completely, accurately, and authoritatively.",
            "3. Maintain clear structure, formatting (markdown headings, bullet points, tables where appropriate).",
            "4. Do not include internal planning jargon (e.g. do not say 'in step 1 we did...'), present the findings directly to the user.",
            "5. If recommendations or next steps are relevant, state them concisely at the conclusion.",
        ]
    )

    return "\n".join(lines)
