"""Live integration tests for Module 18 planning, verification, and replanning components."""

from __future__ import annotations

import os

import pytest
from genai_core import GoogleProvider, LLMService

from plan_and_execute_research_assistant.core.config import AssistantConfig
from plan_and_execute_research_assistant.models import (
    Plan,
    PlanStep,
    RunBudget,
    StepStatus,
)
from plan_and_execute_research_assistant.planning import PlanGenerator, PlanValidator
from plan_and_execute_research_assistant.replanning import ReplanningEngine
from plan_and_execute_research_assistant.verification import StepVerifier


@pytest.fixture
def live_llm_service() -> LLMService:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY / GEMINI_API_KEY not configured for live test.")
    model = os.getenv("DEFAULT_LLM_MODEL", "gemini-3.5-flash-lite")
    provider = GoogleProvider(api_key=api_key, model=model)
    return LLMService(providers={"google": provider}, default_provider="google")


@pytest.mark.asyncio
async def test_live_plan_generator_dag_creation(live_llm_service: LLMService) -> None:
    """Verify live PlanGenerator decomposes a complex research goal into a structurally verified DAG."""
    config = AssistantConfig(default_model="gemini-3.5-flash-lite")
    generator = PlanGenerator(llm_service=live_llm_service, config=config)

    goal = "Investigate account C-101, check billing tickets, and draft refund recommendation"
    plan = await generator.generate_plan(goal=goal, user_id="test_user")

    assert plan.version == 1
    assert len(plan.steps) >= 2
    # Validate the generated plan DAG using 3-state DFS cycle detection
    validator = PlanValidator()
    validator.validate_plan(plan)

    step_ids = {s.id for s in plan.steps}
    for s in plan.steps:
        if s.depends_on:
            for dep in s.depends_on:
                assert dep in step_ids, f"Dependency '{dep}' must exist in plan"


@pytest.mark.asyncio
async def test_live_step_verifier_model_judge(live_llm_service: LLMService) -> None:
    """Verify StepVerifier executes live model-based judge and scores criteria satisfaction."""
    config = AssistantConfig(default_model="gemini-3.5-flash-lite")
    verifier = StepVerifier(llm_service=live_llm_service, config=config)

    step = PlanStep(
        id="step_1",
        description="Lookup customer account C-101 details and current balance",
        success_criteria="Output must contain customer C-101 account tier and current balance in USD",
    )
    deliverable = (
        "Customer profile for C-101 (Alice Johnson) retrieved. Account tier: Gold, current balance: $100.00 USD. "
        "Account is in good standing with zero risk flags."
    )

    result = await verifier.verify_step(
        step=step, output_text=deliverable, execution_status=StepStatus.COMPLETED
    )

    assert result.passed is True
    assert result.programmatic_passed is True
    assert result.model_passed is True
    assert result.score >= 0.85
    assert len(result.reasoning) > 10


@pytest.mark.asyncio
async def test_live_replanning_engine_recovery(live_llm_service: LLMService) -> None:
    """Verify live ReplanningEngine synthesizes replacement steps while preserving completed work."""
    config = AssistantConfig(default_model="gemini-3.5-flash-lite")
    replanner = ReplanningEngine(llm_service=live_llm_service, config=config)

    initial_plan = Plan(
        goal="Diagnose customer churn and recommend retention incentive",
        steps=[
            PlanStep(
                id="step_1",
                description="Fetch customer C-101 account tier",
                status=StepStatus.COMPLETED,
                result_summary="Customer C-101 is Gold tier with 1 active ticket.",
                success_criteria="Account tier retrieved",
            ),
            PlanStep(
                id="step_2",
                description="Call deprecated external scoring service endpoint",
                depends_on=["step_1"],
                status=StepStatus.FAILED,
                error_message="HTTP 404 Service Endpoint Deprecated",
                success_criteria="External churn score obtained",
            ),
        ],
    )

    budget = RunBudget()
    budget.start_timer()

    replan_res = await replanner.replan(
        current_plan=initial_plan,
        failed_step_id="step_2",
        failure_reason="HTTP 404: External churn score service endpoint is permanently deprecated.",
        step_results={"step_1": "Customer C-101 is Gold tier with 1 active ticket."},
        replan_count=0,
        capabilities={
            "search_knowledge_base": "Search internal policy docs",
            "ticket_status": "Check ticket history",
        },
        budget=budget,
    )

    assert replan_res.replan_count == 1
    new_plan = replan_res.new_plan
    assert new_plan.version == 2
    assert new_plan.parent_version == 1

    # Work preservation invariant: step_1 must be preserved and completed
    step_1_candidates = [s for s in new_plan.steps if s.id == "step_1"]
    assert len(step_1_candidates) == 1
    assert step_1_candidates[0].status == StepStatus.COMPLETED

    # Ensure validator passes the new DAG
    PlanValidator().validate_plan(new_plan)
