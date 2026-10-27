"""Prompts for step verification and judge evaluation."""

from __future__ import annotations


def build_judge_prompt(
    step_description: str,
    success_criteria: str,
    output_text: str,
    tool_hint: str | None = None,
) -> str:
    """Build structured evaluation prompt for the LLM judge."""
    lines: list[str] = [
        (
            "You are an impartial, rigorous evaluation judge verifying whether an autonomous AI agent "
            "successfully fulfilled an individual step in an execution plan."
        ),
        "",
        "STEP SPECIFICATION:",
        f"- Objective: {step_description}",
        f"- Verifiable Success Criteria: {success_criteria}",
    ]

    if tool_hint:
        lines.append(f"- Tool / Capability Hint: {tool_hint}")

    lines.extend(
        [
            "",
            "ACTUAL DELIVERABLE PRODUCED BY AGENT:",
            output_text.strip(),
            "",
            "EVALUATION INSTRUCTIONS:",
            "1. Evaluate whether the actual deliverable strictly satisfies every requirement in the success criteria.",
            "2. Score the deliverable on a scale from 0.0 (complete failure/empty) to 1.0 (flawless satisfaction).",
            "3. If any required information, action, calculation, or synthesis is missing or incomplete, set is_satisfied=False and list the missing items in 'missing_criteria'.",
            "4. Provide a clear, objective reasoning for your verdict.",
            "5. Return your evaluation strictly structured according to the StepJudgeVerdict schema.",
        ]
    )

    return "\n".join(lines)
