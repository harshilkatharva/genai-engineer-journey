"""Prompt templates for structured plan generation and repair."""

from __future__ import annotations

from typing import Any

from ..models.plan import Plan


def build_planning_prompt(
    goal: str,
    available_capabilities: dict[str, str] | None = None,
    user_preferences: list[str] | None = None,
    past_plans: list[dict[str, Any]] | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    """Construct the structured planning prompt for LLM decomposition into a DAG."""
    lines: list[str] = [
        (
            "You are an expert AI planning agent. Your role is to decompose the user's high-level goal into an explicit, "
            "executable Directed Acyclic Graph (DAG) of actionable steps."
        ),
        "",
        f"GOAL TO ACHIEVE:\n{goal}",
        "",
    ]

    # 1. Dynamically discovered capabilities
    if available_capabilities:
        lines.append("AVAILABLE EXECUTION CAPABILITIES:")
        for name, desc in sorted(available_capabilities.items()):
            lines.append(f"- {name}: {desc}")
        lines.append(
            "\nNOTE: Assign 'tool_name_hint' in each step to the best matching capability name above, "
            "or null if the step relies strictly on reasoning, analysis, or final synthesis."
        )
        lines.append("")

    # 2. Recalled long-term preferences
    if user_preferences:
        lines.append("USER PREFERENCES & CONSTRAINTS (from long-term memory):")
        for pref in user_preferences:
            lines.append(f"- {pref}")
        lines.append("")

    # 3. Recalled past plan episodes (few-shot reference blueprints)
    if past_plans:
        lines.append("RELEVANT PAST PLAN Blueprints (for reference/adaptation):")
        for i, past in enumerate(past_plans, 1):
            past_goal = past.get("goal", "Past Goal")
            past_steps = past.get("steps", [])
            lines.append(f"Past Case {i} Goal: {past_goal}")
            if past_steps:
                lines.append(f"Steps: {'; '.join(past_steps)}")
            lines.append(f"Outcome: {past.get('result_summary', 'Success')}")
        lines.append("")

    # 4. Context or assumptions
    if context:
        lines.append("ADDITIONAL CONTEXT & INPUTS:")
        for k, v in context.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    # 5. Core structural requirements
    lines.extend(
        [
            "CRITICAL STRUCTURAL REQUIREMENTS (DAG INVARIANTS):",
            "1. Plan must contain at least one step.",
            "2. Each step must have a unique 'id' (e.g. 'step_1', 'step_2').",
            "3. Dependencies must be explicit: 'depends_on' should list IDs of prerequisite steps that must finish first.",
            "4. Independent steps must have 'depends_on': null or [] so they can run concurrently.",
            "5. NO CYCLES: The dependency graph must be a strictly valid Directed Acyclic Graph (DAG).",
            "6. Every step must define concrete, measurable 'success_criteria' to be verified programmatically and by an evaluation judge.",
            "7. Ensure at least one entry step has no dependencies.",
            "8. Specificity: Ensure step descriptions explicitly preserve entity identifiers (e.g. customer IDs like 'C-101', ticket IDs like 'T-101') from the goal.",
            "",
            "Generate the complete structured Plan adhering to the Plan schema.",
        ]
    )

    return "\n".join(lines)


def build_repair_prompt(
    goal: str,
    invalid_plan: Plan | dict[str, Any] | str,
    validation_error: str,
    available_capabilities: dict[str, str] | None = None,
) -> str:
    """Construct a repair prompt feeding validation errors back to the LLM to fix the DAG."""
    plan_str = (
        invalid_plan.model_dump_json(indent=2)
        if isinstance(invalid_plan, Plan)
        else str(invalid_plan)
    )

    lines: list[str] = [
        "Your previous plan generation failed structural DAG validation and must be repaired.",
        "",
        f"ORIGINAL GOAL:\n{goal}",
        "",
        f"VALIDATION ERROR:\n{validation_error}",
        "",
        f"INVALID PLAN ATTEMPT:\n{plan_str}",
        "",
        "REPAIR INSTRUCTIONS:",
        "1. Fix the specific error identified above (resolve circular dependencies, fix missing dependencies, eliminate duplicate IDs).",
        "2. Ensure the revised steps form a strictly valid Directed Acyclic Graph (DAG) with zero cycles.",
        "3. Ensure all step IDs listed in 'depends_on' exist in the plan.",
        "4. Preserve all valid steps and the overall objective.",
        "5. Return a fully corrected Plan conforming to the Plan schema.",
    ]

    if available_capabilities:
        lines.append("")
        lines.append("AVAILABLE CAPABILITIES FOR TOOL HINTS:")
        for name, desc in sorted(available_capabilities.items()):
            lines.append(f"- {name}: {desc}")

    return "\n".join(lines)
