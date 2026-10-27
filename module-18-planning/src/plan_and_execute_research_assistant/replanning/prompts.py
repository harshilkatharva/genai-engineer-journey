"""Prompts for dynamic replanning and failure recovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import PlanStep


def build_replanning_prompt(
    goal: str,
    completed_steps: list[PlanStep],
    failed_step: PlanStep,
    failure_reason: str,
    remaining_steps: list[PlanStep],
    capabilities: dict[str, str] | None = None,
    user_preferences: list[str] | None = None,
) -> str:
    """Construct structured prompt for dynamic subgraph replanning after a step failure."""
    lines: list[str] = [
        "You are an expert AI execution replanner and resilient autonomous problem solver.",
        "An execution plan step has failed after exhausting local retries.",
        "Your task is to revise the execution plan DAG so the overall goal can be fulfilled.",
        "",
        "CRITICAL WORK PRESERVATION INVARIANT:",
        "- All COMPLETED steps are immutable. Do NOT drop, modify, or repeat them.",
        "- Do NOT change the step IDs of completed steps.",
        "- Formulate new or replacement steps that build upon completed work to achieve the goal.",
        "",
        f"TOP-LEVEL GOAL: {goal}",
    ]

    if user_preferences:
        lines.append("")
        lines.append("USER PREFERENCES & CONSTRAINTS:")
        for pref in user_preferences:
            lines.append(f"- {pref}")

    if capabilities:
        lines.append("")
        lines.append("AVAILABLE CAPABILITIES & TOOLS:")
        for name, desc in capabilities.items():
            lines.append(f"- {name}: {desc}")

    lines.append("")
    lines.append("IMMUTABLE COMPLETED STEPS:")
    if completed_steps:
        for step in completed_steps:
            summary = step.result_summary or "Completed successfully."
            lines.append(f"- [{step.id}] {step.description}")
            lines.append(f"  Verified Output: {summary}")
    else:
        lines.append("- None (Failure occurred on initial step)")

    lines.append("")
    lines.append("FAILED STEP DETAILS:")
    lines.append(f"- Step ID: {failed_step.id}")
    lines.append(f"- Description: {failed_step.description}")
    lines.append(f"- Success Criteria: {failed_step.success_criteria}")
    lines.append(f"- Failure Critique / Reason: {failure_reason}")

    if remaining_steps:
        lines.append("")
        lines.append("ORIGINAL REMAINING UNFULFILLED STEPS:")
        for step in remaining_steps:
            deps = f" (depends on: {', '.join(step.depends_on)})" if step.depends_on else ""
            lines.append(f"- [{step.id}] {step.description}{deps}")

    lines.extend(
        [
            "",
            "REPLANNING INSTRUCTIONS:",
            "1. Analyze why the failed step failed and design an alternative or remedial strategy.",
            "2. Retain all COMPLETED steps with their exact IDs, descriptions, and verified outputs.",
            "3. Replace the failed step and adapt subsequent dependent steps to achieve the overall goal.",
            "4. Ensure every newly introduced step has a unique id (e.g., 'step_3_alt', 'remedial_step_4').",
            "5. Ensure the dependency graph is a valid Directed Acyclic Graph (DAG) with zero cycles.",
            "6. Every step must have explicit, verifiable 'success_criteria' and an appropriate 'tool_name_hint'.",
            "7. Populate 'change_summary' detailing the rationale for the structural revisions.",
            "8. Return the complete updated Plan strictly conforming to the Plan schema.",
        ]
    )

    return "\n".join(lines)
