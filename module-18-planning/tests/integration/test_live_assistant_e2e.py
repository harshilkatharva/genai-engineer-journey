"""Live end-to-end integration tests for PlanAndExecuteAssistant with real Gemini & PostgreSQL."""

from __future__ import annotations

import os

import psycopg
import pytest
from genai_core import (
    EmbeddingService,
    GoogleProvider,
    LLMService,
    LocalEmbeddingProvider,
    PGVectorStoreAdapter,
    PostgresMemoryStore,
    RetrievalService,
)
from genai_core.services.memory_service import MemoryService
from research_and_support_agent.agent import BoundedAgent
from research_and_support_agent.core.config import AgentConfig
from research_and_support_agent.data.db_store import (
    load_customer_accounts,
    load_support_tickets,
)
from research_and_support_agent.hitl.postgres_store import PostgresApprovalStore
from research_and_support_agent.memory.manager import AgentMemoryManager

from plan_and_execute_research_assistant.core.config import AssistantConfig
from plan_and_execute_research_assistant.execution import Module17StepExecutor
from plan_and_execute_research_assistant.memory import PlanningMemoryManager
from plan_and_execute_research_assistant.models import (
    PlanReviewDecision,
    PlanReviewRequest,
    ReviewAction,
    RunBudget,
    RunStatus,
    StepStatus,
)
from plan_and_execute_research_assistant.orchestration import PlanAndExecuteAssistant
from plan_and_execute_research_assistant.planning import PlanGenerator, PlanValidator
from plan_and_execute_research_assistant.replanning import ReplanningEngine
from plan_and_execute_research_assistant.review import HumanPlanReviewGateway
from plan_and_execute_research_assistant.scheduling import DAGScheduler
from plan_and_execute_research_assistant.verification import StepVerifier


@pytest.fixture
async def live_assistant() -> PlanAndExecuteAssistant:
    """Fixture providing a fully wired PlanAndExecuteAssistant connected to live Gemini & PostgreSQL."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY / GEMINI_API_KEY not configured for live test.")
    db_url = os.getenv(
        "DATABASE_CONNECTION_CONVERSATION_URL",
        "postgresql://postgres:postgres@127.0.0.1:5432/ai_search",
    )
    model = os.getenv("DEFAULT_LLM_MODEL", "gemini-3.5-flash-lite")

    # 1. Initialize live GenAI Core LLM & Memory
    provider = GoogleProvider(api_key=api_key, model=model)
    llm = LLMService(providers={"google": provider}, default_provider="google")
    emb = EmbeddingService(
        provider=LocalEmbeddingProvider(model="sentence-transformers/all-MiniLM-L6-v2")
    )
    mem_store = PostgresMemoryStore(connection_string=db_url, dimensions=384)
    memory_service = MemoryService(store=mem_store, embedding_service=emb)

    # 2. Initialize live Module 17 BoundedAgent
    accounts = await load_customer_accounts(db_url)
    tickets = await load_support_tickets(db_url)
    approval_store = PostgresApprovalStore(connection_string=db_url)
    pg_vector = PGVectorStoreAdapter(
        dimensions=384,
        connection_factory=lambda: psycopg.AsyncConnection.connect(db_url),
        table_name="document_chunks",
    )
    retrieval = RetrievalService(store=pg_vector, embedding_service=emb)
    agent_mem_mgr = AgentMemoryManager(memory_service=memory_service)
    bounded_agent = BoundedAgent(
        llm_service=llm,
        memory_manager=agent_mem_mgr,
        retrieval_service=retrieval,
        approval_store=approval_store,
        account_store=accounts,
        ticket_store=tickets,
        config=AgentConfig(default_model=model, default_max_steps=5),
    )

    # 3. Initialize Module 18 components
    config = AssistantConfig(
        default_model=model,
        max_concurrency=2,
        auto_approve_review=True,  # auto-approve by default in tests
    )
    executor = Module17StepExecutor(agent=bounded_agent, config=config)
    planning_mem = PlanningMemoryManager(memory_service=memory_service, config=config)
    plan_generator = PlanGenerator(
        llm_service=llm,
        validator=PlanValidator(),
        memory_manager=planning_mem,
        step_executor=executor,
        config=config,
    )
    verifier = StepVerifier(llm_service=llm, config=config)
    replanner = ReplanningEngine(llm_service=llm, config=config)
    scheduler = DAGScheduler(
        executor=executor,
        verifier=verifier,
        config=config,
    )

    return PlanAndExecuteAssistant(
        llm_service=llm,
        executor=executor,
        memory_manager=planning_mem,
        scheduler=scheduler,
        replanner=replanner,
        plan_generator=plan_generator,
        verifier=verifier,
        config=config,
    )


@pytest.mark.asyncio
async def test_live_plan_and_execute_assistant_full_run_success(
    live_assistant: PlanAndExecuteAssistant,
) -> None:
    """Verify live PlanAndExecuteAssistant executes multi-step plan, dual verification, and synthesis."""
    goal = "Retrieve account details for customer C-101 and status for ticket T-101, then formulate a summary"
    user_id = "integration_test_user_1"

    result = await live_assistant.run(goal=goal, user_id=user_id)

    assert result.status == RunStatus.COMPLETED
    assert result.final_deliverable is not None
    assert len(result.final_deliverable) > 30

    # Verify deliverable content mentions retrieved entities
    text_lower = result.final_deliverable.lower()
    assert "c-101" in text_lower or "gold" in text_lower or "100" in text_lower
    assert "t-101" in text_lower or "billing" in text_lower or "ticket" in text_lower

    # Verify plan execution metrics
    assert len(result.plan.steps) >= 2
    for step in result.plan.steps:
        assert step.status == StepStatus.COMPLETED
        assert step.result_summary is not None
        # Verify step execution telemetry and verification in run record
        exec_records = result.run_record.step_executions.get(step.id, [])
        assert len(exec_records) >= 1
        last_exec = exec_records[-1]
        assert last_exec.verification is not None
        assert last_exec.verification.passed is True

    # Verify budget and token tracking
    assert result.total_tokens > 0
    assert result.total_cost_usd >= 0.0
    assert result.duration_ms > 0

    # Verify episodic memory recorded the successful run
    episodes = await live_assistant.memory_manager.memory_service.episodic.get_episodes(
        user_id=user_id, limit=5
    )
    assert len(episodes) > 0


@pytest.mark.asyncio
async def test_live_assistant_human_review_rejection(
    live_assistant: PlanAndExecuteAssistant,
) -> None:
    """Verify human plan review rejection correctly cancels execution without executing steps."""
    goal = "Perform high-risk batch database operation on all customer accounts"
    user_id = "integration_test_user_2"

    async def reject_provider(request: PlanReviewRequest) -> PlanReviewDecision:
        return PlanReviewDecision(
            action=ReviewAction.REJECT,
            reviewer_id="human_reviewer_test",
            reason="Operation rejected: batch operations require manual DBA intervention.",
        )

    # Enable review gateway with auto_approve_review=False
    live_assistant.review_gateway = HumanPlanReviewGateway(
        config=AssistantConfig(auto_approve_review=False)
    )

    result = await live_assistant.run(
        goal=goal,
        user_id=user_id,
        decision_provider=reject_provider,
    )

    assert result.status == RunStatus.CANCELLED
    assert result.error_message is not None
    assert "rejected" in result.error_message.lower()
    # Ensure zero steps were executed
    for step in result.plan.steps:
        assert step.status in (StepStatus.PENDING, StepStatus.SKIPPED)


@pytest.mark.asyncio
async def test_live_assistant_budget_enforcement(
    live_assistant: PlanAndExecuteAssistant,
) -> None:
    """Verify RunBudget halts execution when wall-clock timeout or token ceiling is breached."""
    goal = "Retrieve account details for customer C-101 and status for ticket T-101, then formulate a summary"
    user_id = "integration_test_user_3"

    # Set strict budget with near-zero timeout to trigger budget exhaustion
    budget = RunBudget(
        total_timeout_seconds=0.001,
    )

    result = await live_assistant.run(goal=goal, user_id=user_id, budget=budget)

    assert result.status == RunStatus.FAILED
    assert result.error_message is not None
    assert (
        "budget exceeded" in result.error_message.lower()
        or "timeout" in result.error_message.lower()
    )
