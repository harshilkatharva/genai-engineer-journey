"""Unit tests for StepVerifier and dual verification gates."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from genai_core import LLMManagerResponse, LLMService

from plan_and_execute_research_assistant.core import AssistantConfig
from plan_and_execute_research_assistant.models import (
    PlanStep,
    StepJudgeVerdict,
    StepStatus,
)
from plan_and_execute_research_assistant.verification import (
    StepVerifier,
    build_judge_prompt,
)


@pytest.fixture
def sample_step() -> PlanStep:
    """Fixture providing a typical plan step."""
    return PlanStep(
        id="step_audit_1",
        description="Extract support ticket resolution times and categorize by priority",
        tool_name_hint="ticket_analytics",
        success_criteria="Summary includes average resolution time per priority tier",
    )


@pytest.fixture
def mock_llm_service() -> MagicMock:
    """Mock LLMService."""
    svc = MagicMock(spec=LLMService)
    svc.complete_structure = AsyncMock()
    return svc


def test_build_judge_prompt(sample_step: PlanStep) -> None:
    """Verify build_judge_prompt includes objective, criteria, and agent deliverable."""
    prompt = build_judge_prompt(
        step_description=sample_step.description,
        success_criteria=sample_step.success_criteria,
        output_text="Priority P1: 4.2h, Priority P2: 12.8h, Priority P3: 28.5h",
        tool_hint=sample_step.tool_name_hint,
    )
    assert "Extract support ticket resolution times" in prompt
    assert "average resolution time per priority tier" in prompt
    assert "<deliverable>\nPriority P1: 4.2h" in prompt
    assert "</deliverable>" in prompt
    assert "ticket_analytics" in prompt


def test_programmatic_verification_success(
    sample_step: PlanStep, mock_llm_service: MagicMock
) -> None:
    """Verify programmatic check passes on valid completed output >= 20 chars."""
    verifier = StepVerifier(llm_service=mock_llm_service)
    passed, msg = verifier.verify_programmatic(
        step=sample_step,
        output_text="P1 average: 4.2h. P2 average: 12.8h. P3 average: 28.5h. All tickets audited.",
        execution_status=StepStatus.COMPLETED,
    )
    assert passed is True
    assert "passed" in msg


def test_programmatic_verification_failed_status(
    sample_step: PlanStep, mock_llm_service: MagicMock
) -> None:
    """Verify programmatic check fails if step status is not COMPLETED."""
    verifier = StepVerifier(llm_service=mock_llm_service)
    passed, msg = verifier.verify_programmatic(
        step=sample_step,
        output_text="Valid output text that exceeds twenty characters easily.",
        execution_status=StepStatus.FAILED,
    )
    assert passed is False
    assert "failed with status" in msg


def test_programmatic_verification_empty_text(
    sample_step: PlanStep, mock_llm_service: MagicMock
) -> None:
    """Verify programmatic check fails on empty output text."""
    verifier = StepVerifier(llm_service=mock_llm_service)
    passed, msg = verifier.verify_programmatic(
        step=sample_step,
        output_text="   ",
        execution_status=StepStatus.COMPLETED,
    )
    assert passed is False
    assert "empty" in msg


def test_programmatic_verification_too_short(
    sample_step: PlanStep, mock_llm_service: MagicMock
) -> None:
    """Verify programmatic check fails if deliverable is under 20 characters."""
    verifier = StepVerifier(llm_service=mock_llm_service)
    passed, msg = verifier.verify_programmatic(
        step=sample_step,
        output_text="Done.",
        execution_status=StepStatus.COMPLETED,
    )
    assert passed is False
    assert "too short" in msg


@pytest.mark.asyncio
async def test_verify_step_happy_path(sample_step: PlanStep, mock_llm_service: MagicMock) -> None:
    """Verify dual gates pass when programmatic checks and model judge succeed."""
    verdict = StepJudgeVerdict(
        is_satisfied=True,
        score=0.92,
        reasoning="All priority tiers are clearly reported with average times.",
        missing_criteria=[],
    )
    mock_llm_service.complete_structure.return_value = LLMManagerResponse(
        structured_response=verdict,
        model="gemini-3.5-flash-lite",
    )

    verifier = StepVerifier(llm_service=mock_llm_service)
    result = await verifier.verify_step(
        step=sample_step,
        output_text="Priority P1: 4.2h, Priority P2: 12.8h, Priority P3: 28.5h across 150 tickets.",
        execution_status=StepStatus.COMPLETED,
    )

    assert result.passed is True
    assert result.programmatic_passed is True
    assert result.model_passed is True
    assert result.score == 0.92
    assert "priority tiers" in result.reasoning


@pytest.mark.asyncio
async def test_verify_step_model_score_below_threshold(
    sample_step: PlanStep, mock_llm_service: MagicMock
) -> None:
    """Verify step fails if model judge score is below threshold (0.85)."""
    verdict = StepJudgeVerdict(
        is_satisfied=True,
        score=0.75,
        reasoning="P1 and P2 are reported, but P3 priority was omitted.",
        missing_criteria=["Missing P3 resolution time"],
    )
    mock_llm_service.complete_structure.return_value = LLMManagerResponse(
        structured_response=verdict,
        model="gemini-3.5-flash-lite",
    )

    config = AssistantConfig(verification_threshold=0.85)
    verifier = StepVerifier(llm_service=mock_llm_service, config=config)
    result = await verifier.verify_step(
        step=sample_step,
        output_text="Priority P1: 4.2h, Priority P2: 12.8h.",
        execution_status=StepStatus.COMPLETED,
    )

    assert result.passed is False
    assert result.programmatic_passed is True
    assert result.model_passed is False
    assert result.score == 0.75
    assert "Missing P3" in result.missing_criteria[0]


@pytest.mark.asyncio
async def test_verify_step_model_unsatisfied(
    sample_step: PlanStep, mock_llm_service: MagicMock
) -> None:
    """Verify step fails if model judge returns is_satisfied=False even if score is high."""
    verdict = StepJudgeVerdict(
        is_satisfied=False,
        score=0.88,
        reasoning="Output formatted nicely but ticket counts are unverified.",
        missing_criteria=["Unverified ticket count source"],
    )
    mock_llm_service.complete_structure.return_value = LLMManagerResponse(
        structured_response=verdict,
        model="gemini-3.5-flash-lite",
    )

    verifier = StepVerifier(llm_service=mock_llm_service)
    result = await verifier.verify_step(
        step=sample_step,
        output_text="Priority P1: 4.2h, Priority P2: 12.8h, Priority P3: 28.5h.",
        execution_status=StepStatus.COMPLETED,
    )

    assert result.passed is False
    assert result.model_passed is False


@pytest.mark.asyncio
async def test_verify_step_programmatic_failure_bypasses_model(
    sample_step: PlanStep, mock_llm_service: MagicMock
) -> None:
    """Verify programmatic failure short-circuits without calling model judge."""
    verifier = StepVerifier(llm_service=mock_llm_service)
    result = await verifier.verify_step(
        step=sample_step,
        output_text="Too short",
        execution_status=StepStatus.COMPLETED,
    )

    assert result.passed is False
    assert result.programmatic_passed is False
    mock_llm_service.complete_structure.assert_not_called()


@pytest.mark.asyncio
async def test_verify_step_llm_exception_handled_gracefully(
    sample_step: PlanStep, mock_llm_service: MagicMock
) -> None:
    """Verify LLMService error during judge evaluation is caught and returns failing result."""
    mock_llm_service.complete_structure.side_effect = RuntimeError("Judge API quota exceeded")

    verifier = StepVerifier(llm_service=mock_llm_service)
    result = await verifier.verify_step(
        step=sample_step,
        output_text="Valid length output text that passes programmatic assertions.",
        execution_status=StepStatus.COMPLETED,
    )

    assert result.passed is False
    assert result.model_passed is False
    assert "Judge API quota exceeded" in result.reasoning


def test_build_judge_prompt_includes_negative_findings_instruction(
    sample_step: PlanStep,
) -> None:
    """Verify build_judge_prompt contains explicit rules for negative findings & resource absence."""
    prompt = build_judge_prompt(
        step_description=sample_step.description,
        success_criteria=sample_step.success_criteria,
        output_text="Requested record was not found in database.",
        tool_hint=sample_step.tool_name_hint,
    )
    assert "HANDLING NEGATIVE FINDINGS & RESOURCE ABSENCE" in prompt
    assert "VALID AND SUCCESSFUL FACTUAL FINDING" in prompt
    assert "Do NOT penalize the agent" in prompt
    assert "not found" in prompt


@pytest.mark.asyncio
async def test_verify_step_negative_finding_verified(
    mock_llm_service: MagicMock,
) -> None:
    """Verify step passes when tool returns negative finding and judge confirms absence."""
    step = PlanStep(
        id="step_lookup_absent",
        description="Lookup status for record R-999",
        tool_name_hint="record_status",
        success_criteria="Record R-999 details are retrieved",
    )
    verdict = StepJudgeVerdict(
        is_satisfied=True,
        score=0.95,
        reasoning="The agent queried record_status and conclusively confirmed R-999 does not exist in system records.",
        missing_criteria=[],
    )
    mock_llm_service.complete_structure.return_value = LLMManagerResponse(
        structured_response=verdict,
        model="gemini-3.5-flash-lite",
    )

    verifier = StepVerifier(llm_service=mock_llm_service)
    result = await verifier.verify_step(
        step=step,
        output_text="Record R-999 could not be found in the system after querying record_status.",
        execution_status=StepStatus.COMPLETED,
    )

    assert result.passed is True
    assert result.programmatic_passed is True
    assert result.model_passed is True
    assert result.score == 0.95
    assert "conclusively confirmed R-999 does not exist" in result.reasoning


@pytest.mark.asyncio
async def test_verify_step_approval_required_verified(
    mock_llm_service: MagicMock,
) -> None:
    """Verify step passes when tool initiates irreversible action paused for human approval."""
    step = PlanStep(
        id="step_refund",
        description="Execute support refund of $50.00 USD for customer C-102",
        tool_name_hint="execute_refund",
        success_criteria="Refund successfully executed for customer C-102",
    )
    verdict = StepJudgeVerdict(
        is_satisfied=True,
        score=0.92,
        reasoning="The agent initiated execute_refund and properly registered it for human authorization per security policy.",
        missing_criteria=[],
    )
    mock_llm_service.complete_structure.return_value = LLMManagerResponse(
        structured_response=verdict,
        model="gemini-3.5-flash-lite",
    )

    deliverable = (
        "Action 'execute_refund' (irreversible) was successfully initiated and paused for human authorization.\n"
        "- Approval Request ID: req_9999\n"
        "- Parameters: {'customer_id': 'C-102', 'amount_usd': 50.0}\n"
        "- Safeguard Reason: Refund exceeds automated threshold\n"
        "The operation is safely registered in the approval store awaiting reviewer sign-off."
    )

    verifier = StepVerifier(llm_service=mock_llm_service)
    result = await verifier.verify_step(
        step=step,
        output_text=deliverable,
        execution_status=StepStatus.COMPLETED,
    )

    assert result.passed is True
    assert result.programmatic_passed is True
    assert result.model_passed is True
    assert result.score == 0.92
    assert "properly registered it for human authorization" in result.reasoning


@pytest.mark.asyncio
async def test_verifier_tracks_tokens_and_cost(mock_llm_service: MagicMock) -> None:
    """Verify StepVerifier extracts usage metrics from LLM response and populates StepVerificationResult."""
    step = PlanStep(
        id="step_tokens",
        description="Verify pricing metrics",
        success_criteria="Metrics verified",
    )
    verdict = StepJudgeVerdict(
        is_satisfied=True,
        score=0.90,
        reasoning="Metrics verified completely.",
        missing_criteria=[],
    )
    mock_response = MagicMock()
    mock_response.structured_response = verdict
    mock_response.model = "gemini-3.5-flash-lite"
    mock_response.usage = {"input_tokens": 300, "output_tokens": 150, "total_tokens": 450}
    mock_response.cost_usd = 0.00025
    mock_llm_service.complete_structure.return_value = mock_response

    verifier = StepVerifier(llm_service=mock_llm_service)
    result = await verifier.verify_step(
        step=step,
        output_text="Pricing metrics: compute is $0.50/hr and storage is $0.02/GB.",
        execution_status=StepStatus.COMPLETED,
    )

    assert result.passed is True
    assert result.tokens_used == 450
    assert result.cost_usd == 0.00025
