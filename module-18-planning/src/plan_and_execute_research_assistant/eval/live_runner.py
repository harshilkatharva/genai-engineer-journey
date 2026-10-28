"""Live benchmark execution targets using real Gemini API and PostgreSQL."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import psycopg
from genai_core import (
    EmbeddingService,
    GoogleProvider,
    LLMService,
    LocalEmbeddingProvider,
    PGVectorStoreAdapter,
    PostgresMemoryStore,
    RetrievalService,
    ToolLoopStatus,
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

from ..core import AssistantConfig, get_assistant_config, truncate_to_tokens
from ..execution import Module17StepExecutor
from ..memory import PlanningMemoryManager
from ..models import RunBudget, RunStatus, StepStatus
from ..orchestration import PlanAndExecuteAssistant
from ..planning import PlanGenerator, PlanValidator
from ..replanning import ReplanningEngine
from ..scheduling import DAGScheduler
from ..verification import StepVerifier
from .metrics import SystemEvalResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from genai_core.models.evaluation_model import EvalSample

_logger = logging.getLogger("plan_and_execute_research_assistant.eval.live_runner")


async def create_live_benchmark_targets(
    api_key: str,
    db_url: str = "postgresql://postgres:postgres@127.0.0.1:5432/ai_search",
    model: str | None = None,
    config: AssistantConfig | None = None,
) -> tuple[
    Callable[[EvalSample], Awaitable[SystemEvalResult]],
    Callable[[EvalSample], Awaitable[SystemEvalResult]],
]:
    """Construct live execution targets for comparative evaluation.

    Args:
        api_key: Google Gemini API key.
        db_url: PostgreSQL connection URL.
        model: Optional model identifier override (falls back to config.default_model).
        config: Optional AssistantConfig instance (falls back to get_assistant_config()).

    Returns:
        Tuple of (live_planned_target, live_reactive_target).
    """
    effective_config = config or get_assistant_config()
    target_model = model or effective_config.default_model

    provider = GoogleProvider(api_key=api_key, model=target_model)
    llm = LLMService(providers={"google": provider}, default_provider="google")
    emb = EmbeddingService(
        provider=LocalEmbeddingProvider(model="sentence-transformers/all-MiniLM-L6-v2")
    )
    mem_store = PostgresMemoryStore(connection_string=db_url, dimensions=384)
    memory_service = MemoryService(store=mem_store, embedding_service=emb)

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

    web_limit_chars = effective_config.max_web_retrieved_tokens * 4
    agent_config = AgentConfig(
        default_model=target_model,
        default_max_steps=effective_config.default_step_max_iterations + 1,
        default_timeout_seconds=min(90.0, effective_config.default_timeout_seconds),
        default_max_tokens=effective_config.max_total_tokens,
        default_max_cost_usd=effective_config.max_total_cost_usd,
        web_search_max_chars=web_limit_chars,
    )

    # 1. Module 17 Reactive BoundedAgent
    bounded_agent = BoundedAgent(
        llm_service=llm,
        memory_manager=agent_mem_mgr,
        retrieval_service=retrieval,
        approval_store=approval_store,
        account_store=accounts,
        ticket_store=tickets,
        config=agent_config,
    )

    # Align ToolExecutor max_output_chars with the 5K token limit
    if hasattr(bounded_agent, "executor") and bounded_agent.executor:
        bounded_agent.executor.max_output_chars = web_limit_chars

    # 2. Module 18 Planned Assistant
    if hasattr(bounded_agent, "tool_registry") and bounded_agent.tool_registry:
        if not effective_config.enable_episodic_similarity_search:
            for t_name in [
                "get_episodic_memory",
                "delete_episodic_memory_all",
                "delete_episodic_memory_episodes",
            ]:
                bounded_agent.tool_registry.unregister(t_name)

        # Enforce 5K token limit on web search responses
        sw_tool = bounded_agent.tool_registry.get("search_web")
        if sw_tool is not None:
            orig_run = sw_tool.run

            async def bounded_search_web(*args: Any, **kwargs: Any) -> str:
                raw_res = await orig_run(*args, **kwargs)
                res_str = str(raw_res)
                truncated_res, _ = truncate_to_tokens(
                    text=res_str,
                    max_tokens=effective_config.max_web_retrieved_tokens,
                    suffix=f"\n... [Web-retrieved data truncated to {effective_config.max_web_retrieved_tokens} tokens limit]",
                )
                return truncated_res

            sw_tool.run = bounded_search_web  # type: ignore[method-assign]

    # Ensure benchmark seed memory and ticket for C-104 are present
    try:
        from genai_core.models.memory_model import MemoryCategory

        existing_ltm = await memory_service.long_term.view(user_id="C-104")
        if not existing_ltm:
            await memory_service.remember(
                user_id="C-104",
                content="Customer C-104 requires rapid SLA turnaround, escalation on billing disputes, and proactive communication.",
                category=MemoryCategory.PREFERENCE,
            )
    except Exception as exc:  # noqa: BLE001
        _logger.warning("Could not pre-seed C-104 memory: %s", exc)

    if "T-104" not in tickets:
        from research_and_support_agent.models import SupportTicket, TicketStatus

        tickets["T-104"] = SupportTicket(
            ticket_id="T-104",
            customer_id="C-104",
            title="Repeated billing turnaround delay",
            description="Customer expressed concern regarding turnaround delay and requested immediate SLA review.",
            status=TicketStatus.OPEN,
            assigned_team="tier1_support",
            priority="high",
            history=[{"action": "created", "note": "Billing turnaround ticket opened"}],
        )

    executor = Module17StepExecutor(agent=bounded_agent, config=effective_config)
    planning_mem = PlanningMemoryManager(memory_service=memory_service, config=effective_config)
    plan_generator = PlanGenerator(
        llm_service=llm,
        validator=PlanValidator(),
        memory_manager=planning_mem,
        step_executor=executor,
        config=effective_config,
    )
    verifier = StepVerifier(llm_service=llm, config=effective_config)
    replanner = ReplanningEngine(llm_service=llm, config=effective_config)
    scheduler = DAGScheduler(
        executor=executor,
        verifier=verifier,
        config=effective_config,
    )
    assistant = PlanAndExecuteAssistant(
        llm_service=llm,
        executor=executor,
        memory_manager=planning_mem,
        scheduler=scheduler,
        replanner=replanner,
        plan_generator=plan_generator,
        verifier=verifier,
        config=effective_config,
    )

    async def live_planned_target(sample: EvalSample) -> SystemEvalResult:
        g_id = str(sample.id)
        category = sample.metadata.get("category", "general")
        print(f"  -> [{g_id}] Evaluating Planned Assistant ({category})...", flush=True)
        start_t = time.monotonic()
        try:
            budget = RunBudget(
                max_total_tokens=effective_config.max_total_tokens,
                max_total_cost_usd=effective_config.max_total_cost_usd,
                total_timeout_seconds=effective_config.default_timeout_seconds,
            )
            user_id = sample.metadata.get("user_id") or f"eval_user_{g_id}"
            result = await assistant.run(goal=sample.query, user_id=user_id, budget=budget)
            elapsed = round(time.monotonic() - start_t, 3)

            deliverable = result.final_deliverable or ""
            is_completed = result.status == RunStatus.COMPLETED
            all_steps_ok = (
                all(s.status == StepStatus.COMPLETED for s in result.plan.steps)
                if result.plan.steps
                else False
            )
            req_matched = True
            if sample.expected_answer and sample.expected_answer.required:
                matches = [
                    kw
                    for kw in sample.expected_answer.required
                    if kw.lower() in deliverable.lower()
                ]
                req_matched = len(matches) >= max(1, len(sample.expected_answer.required) // 2)

            passed = is_completed and (all_steps_ok or req_matched) and len(deliverable) > 20
            replan_count = max(0, result.plan.version - 1)
            status_tag = "PASSED" if passed else "FAILED"
            print(
                f"     Planned [{g_id}]: {status_tag} in {elapsed}s ({len(result.plan.steps)} steps, {result.total_tokens} tokens)",
                flush=True,
            )

            return SystemEvalResult(
                system_name="Module 18 Planned Assistant",
                goal_id=g_id,
                category=category,
                passed=passed,
                steps=len(result.plan.steps),
                tokens=result.total_tokens,
                cost_usd=result.total_cost_usd,
                latency_s=elapsed,
                replan_count=replan_count,
                deliverable=deliverable,
                error_message=result.error_message,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = round(time.monotonic() - start_t, 3)
            print(f"     Planned [{g_id}]: ERROR in {elapsed}s ({exc})", flush=True)
            return SystemEvalResult(
                system_name="Module 18 Planned Assistant",
                goal_id=g_id,
                category=category,
                passed=False,
                steps=1,
                tokens=0,
                cost_usd=0.0,
                latency_s=elapsed,
                replan_count=0,
                deliverable="",
                error_message=str(exc),
            )

    async def live_reactive_target(sample: EvalSample) -> SystemEvalResult:
        g_id = str(sample.id)
        category = sample.metadata.get("category", "general")
        print(f"  -> [{g_id}] Evaluating Reactive Baseline ({category})...", flush=True)
        start_t = time.monotonic()
        try:
            user_id = sample.metadata.get("user_id")
            query = sample.query
            if user_id and user_id not in query:
                query = f"{query} (Context: User/Account ID is {user_id})"

            agent_res = await bounded_agent.run(query)
            elapsed = round(time.monotonic() - start_t, 3)

            deliverable = agent_res.final_text or ""
            is_completed = agent_res.status == ToolLoopStatus.COMPLETED
            req_matched = True
            if sample.expected_answer and sample.expected_answer.required:
                matches = [
                    kw
                    for kw in sample.expected_answer.required
                    if kw.lower() in deliverable.lower()
                ]
                req_matched = len(matches) >= max(1, len(sample.expected_answer.required) // 2)

            passed = is_completed and req_matched and len(deliverable) > 20
            status_tag = "PASSED" if passed else "FAILED"
            print(
                f"     Reactive [{g_id}]: {status_tag} in {elapsed}s ({agent_res.iterations} turns, {agent_res.total_tokens} tokens)",
                flush=True,
            )

            return SystemEvalResult(
                system_name="Module 17 Reactive BoundedAgent",
                goal_id=g_id,
                category=category,
                passed=passed,
                steps=agent_res.iterations,
                tokens=agent_res.total_tokens,
                cost_usd=agent_res.total_cost_usd,
                latency_s=elapsed,
                replan_count=0,
                deliverable=deliverable,
                error_message=agent_res.error_message,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = round(time.monotonic() - start_t, 3)
            return SystemEvalResult(
                system_name="Module 17 Reactive BoundedAgent",
                goal_id=g_id,
                category=category,
                passed=False,
                steps=1,
                tokens=0,
                cost_usd=0.0,
                latency_s=elapsed,
                replan_count=0,
                deliverable="",
                error_message=str(exc),
            )

    return live_planned_target, live_reactive_target
