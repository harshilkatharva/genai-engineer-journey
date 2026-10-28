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
            "1. Evaluate whether the actual deliverable satisfies the step objective and verifiable success criteria.",
            (
                "2. HANDLING NEGATIVE FINDINGS & RESOURCE ABSENCE (CRITICAL RULE):\n"
                "   - For inquiry, query, search, retrieval, lookup, or audit steps: if the agent used the appropriate\n"
                "     capability/tool and conclusively established that a requested entity, record, document, identifier,\n"
                "     account, or information does NOT exist, is NOT found (e.g. 404, not found, or empty search results),\n"
                "     or is absent from the system, this is a VALID AND SUCCESSFUL FACTUAL FINDING about the state of the data.\n"
                "   - Do NOT penalize the agent or set is_satisfied=False simply because the target item or record does not exist\n"
                "     in the underlying data store or external source.\n"
                "   - If the deliverable clearly documents the absence, non-existence, or 'not found' status of the\n"
                "     requested item, consider the inquiry objective fulfilled, set is_satisfied=True, and assign\n"
                "     a passing confidence score (>= 0.85).\n"
                "   - Only set is_satisfied=False if the agent encountered an unhandled execution crash or internal error\n"
                "     (e.g., unhandled exception, broken connection, malformed parameters), fabricated fictitious data,\n"
                "     or failed to perform the required inquiry action at all."
            ),
            (
                "3. HANDLING ACTIONS REQUIRING HUMAN AUTHORIZATION (CRITICAL RULE):\n"
                "   - For operational actions (e.g., refund execution, record deletion, sensitive account modification): if the\n"
                "     action was successfully initiated, registered, and safely paused for human reviewer authorization\n"
                "     (e.g., status 'approval_required' or an approval request registered with a request ID),\n"
                "     this represents a VALID AND SUCCESSFUL EXECUTION under enterprise safety policies.\n"
                "   - Do NOT penalize the agent or set is_satisfied=False simply because the operation requires human review before finalizing.\n"
                "   - If the deliverable clearly documents that the action was initiated and is awaiting operator authorization,\n"
                "     consider the initiation objective fulfilled, set is_satisfied=True, and assign a passing confidence score (>= 0.85)."
            ),
            "4. Score the deliverable on a scale from 0.0 (complete failure/empty/unhandled error) to 1.0 (flawless satisfaction or clear verified finding).",
            "5. If any required information, action, calculation, or synthesis is missing due to agent oversight (and not due to confirmed absence in the data source or safety review pause), set is_satisfied=False and list the missing items in 'missing_criteria'.",
            "6. Provide a clear, objective reasoning for your verdict.",
            "7. Return your evaluation strictly structured according to the StepJudgeVerdict schema.",
        ]
    )

    return "\n".join(lines)
