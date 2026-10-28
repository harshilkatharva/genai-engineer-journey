# Module 18: Plan-and-Execute Research Assistant — Goals & Progress

## 1. Project Goal

Build a production-grade, bounded **Plan-and-Execute Research Assistant** that decomposes complex research and operational goals into explicit, validated Directed Acyclic Graph (DAG) plans, schedules independent steps concurrently, executes bounded multi-turn steps through the Module 17 agent package (`research_and_support_agent`), verifies each step through dual programmatic and model-based criteria, supports dynamic replanning upon failure while preserving completed work, integrates with `genai_core` memory systems, enforces run-wide budgets (tokens, cost, deadline), requires human plan review with immutable versioning, and validates against a 10-goal comparative benchmark against the Module 17 reactive baseline.

This planner also serves as the extensible foundation for:
- **Module 19:** Plan reflection and self-critique.
- **Module 20:** Delegating plan steps to specialist agents.

---

## 2. Source of Truth

- `GOAL.md` — Project milestones, completed items, and pending tasks.
- `DECISIONS.md` — Append-only architectural decision log explaining why files, models, and flows were chosen.
- `BUSINESS_LOGIC.md` — Confirmed business rules, DAG validation schemas, budgets, replanning policies, and verification thresholds.

---

## 3. Scope & Milestones

### Milestone 1: Domain Models, DAG Validation & Structural Integrity
- [x] Implement core Pydantic domain models:
  - `PlanStep`, `Plan`, `PlanVersion`, `StepStatus`, `RunStatus`.
  - `PlanReviewRequest`, `PlanReviewDecision`, `ReviewAction`.
  - `StepVerificationResult`, `StepJudgeVerdict`.
  - `RunRecord`, `StepExecutionRecord`, `RunBudget`.
- [x] Implement structural `PlanValidator`:
  - Non-empty plan validation (>= 1 step).
  - Unique step ID validation.
  - Self-dependency and cycle detection using 3-state Depth-First Search (`UNVISITED`, `VISITING`, `VISITED`).
  - Referential integrity check (all dependencies exist).
  - Root step reachability check.
- [x] Add comprehensive unit tests in `tests/unit/planning/test_plan_validator.py` covering valid plans, direct/indirect cycles, duplicate IDs, missing dependencies, and self-dependencies.

### Milestone 2: Structured Plan Generation, Automated Repair & Memory Integration
- [x] Implement `PlanGenerator` utilizing `genai_core.LLMService.complete_structure` with Pydantic `Plan` schema.
- [x] Implement `PlanRepairer` capturing `PlanValidationError` stack traces and re-prompting the LLM with structured critique up to `max_planning_retries` (default: 3).
- [x] Implement `PlanningMemoryManager`:
  - Recall user preferences and domain constraints from `genai_core.LongTermMemory` via `search_longterm()`.
  - Record completed plan episodes in `genai_core.EpisodicMemory` for auditability and idempotency.
  - Support configurable episodic past plan retrieval (`enable_episodic_similarity_search`) with graceful zero-shot fallback.
- [x] Add unit tests in `tests/unit/planning/test_plan_generator.py` and `tests/unit/memory/test_memory_manager.py`.

### Milestone 3: Human Plan Review Gateway & Immutable Plan Versioning
- [x] Implement `HumanPlanReviewGateway`:
  - Present proposed plans (steps, dependencies, criteria) for operator review.
  - Support three operator actions: `APPROVE`, `MODIFY`, `REJECT`.
  - Re-validate modified plans through `PlanValidator`.
  - Provide configurable review bypass (`auto_approve_review=True`) for automated evaluation and CI pipelines.
- [x] Implement immutable plan versioning:
  - Tag initial plan as `version=1`, `parent_version=None`, `created_by="planner_llm"`.
  - Persist human-edited plans as `version=2`, `parent_version=1`, `created_by="human_reviewer"`.
  - Persist dynamic replans as `version=N+1`, `parent_version=N`, `created_by="replanner"`.
- [x] Implement `RunRecordStore` protocol and `FileRunRecordStore` persisting full run lifecycles (`runs/run_<id>.json`).
- [x] Add unit tests in `tests/unit/review/test_human_review.py` and `tests/unit/storage/test_run_records.py`.

### Milestone 4: Async DAG Scheduler, Module 17 Step Execution & Dual Verification
- [x] Implement pure Python `DAGScheduler` using `asyncio`:
  - Continuously resolve dependency readiness: transition steps to `READY` when all predecessor steps are `COMPLETED`.
  - Dispatch ready independent steps concurrently via `asyncio.create_task`.
  - Enforce concurrency cap strictly using `asyncio.Semaphore(max_concurrency)` (default: 3).
  - Aggregate predecessor outputs and inject context into downstream dependent steps.
- [x] Implement `Module17StepExecutor` conforming to `StepExecutor` protocol:
  - Wrap `research_and_support_agent.BoundedAgent`.
  - Package step instructions, predecessor summaries, tool hints, and step-level budgets.
- [x] Implement `StepVerifier` combining two independent gates:
  - **Programmatic Verifier:** Validates completion status, non-empty deliverable, minimum length, and deterministic regex/JSON rules.
  - **Model-Based Judge Verifier:** Evaluates whether actual output satisfies `step.success_criteria` using `LLMService.complete_structure` with `StepJudgeVerdict` (threshold >= 0.85).
- [x] Support step-level retries up to `max_step_retries` (default: 2) with verification feedback before declaring step failure.
- [x] Add unit tests in `tests/unit/scheduling/test_dag_scheduler.py` and `tests/unit/verification/test_verification.py`.

### Milestone 5: Concurrency-Safe Dynamic Replanning & Failure Recovery
- [x] Implement `ReplanningEngine`:
  - Triggered when a step fails after exhausting step-level retries.
  - **Concurrency-Safe Sibling Handling:** Await or settle currently running sibling steps; collect completed outputs into the execution cache; do not cancel or discard valid completed work.
  - **Work Preservation:** Retain all `COMPLETED` steps as immutable dependencies with their cached outputs; never re-execute them.
  - **Subgraph Replanning:** Formulate replanning prompt providing overall goal, completed steps, failed step, and failure/verification critique.
  - Generate replacement/remedial steps for remaining unfulfilled objectives, re-validating the full DAG.
  - Link new plan version (`version=parent_version+1`, `parent_version=current_version`).
  - Enforce global replanning limit (`max_replans=3`) to prevent thrashing.
- [x] Add unit tests in `tests/unit/replanning/test_replanning.py` verifying state isolation, sibling settling, and work preservation.

### Milestone 6: Run-Wide Budget Tracking & Orchestrator Facade
- [x] Implement `RunBudget` tracking aggregate token consumption, total cost ($ USD), and wall-clock deadline across all lifecycle phases (planning, human review, parallel step execution, verification, retries, and replanning).
- [x] Implement `PlanAndExecuteAssistant` facade coordinating initialization, memory recall, planning, review, scheduling, replanning, synthesis, and run persistence.
- [x] Ensure future extensibility contracts (`PlanReflector` protocol for Module 19, `StepExecutor` protocol for Module 20) are cleanly decoupled.
- [x] Add unit tests in `tests/unit/orchestration/test_assistant.py` and `tests/unit/budget/test_run_budget.py`.

### Milestone 7: 10-Goal Fair Comparative Evaluation Suite
- [x] Curate 10 benchmark goals covering multi-source research, complex operations, and error recovery in `eval/dataset.py`.
- [x] Implement runnable evaluation script `scripts/run_eval.py`:
  - Run both candidate (**Module 18 Planned Assistant**) and baseline (**Module 17 Reactive BoundedAgent**) under identical configurations.
  - Enforce equal total token ($40k$) and cost ($0.25$) caps and identical models (`gemini-3.5-flash-lite`).
  - Evaluate final deliverables using identical model-based judge criteria.
  - Report comparative metrics: Completion Rate (%), Steps/Turns, Total Cost ($ USD), Latency (s), and Replanning Count.
  - Support both interactive CLI prompt and non-interactive `--mock` / `--live` modes with `--limit` and `--goal-ids` flags.
- [x] Implement `create_live_benchmark_targets` in `src/plan_and_execute_research_assistant/eval/live_runner.py` wiring real Gemini API (`gemini-3.5-flash-lite`), live PostgreSQL database stores, pgvector retrieval, and real-time terminal progress.
- [x] Implement inter-goal cooldown (30s) in `eval/runner.py` and `--pause-between-goals` CLI flag in `scripts/run_eval.py` to protect against LLM API rate limits while preserving accurate latency measurements.
- [x] Implement real-time step telemetry streaming in `DAGScheduler` to guarantee zero record loss upon budget exhaustion, and bound all web-retrieved data to 5,000 tokens using `tiktoken` in `Module17StepExecutor` and benchmark tool registration.
- [x] Add integration tests in `tests/integration/test_eval_benchmark.py` (gated by `INTEGRATION_TEST=1`).

### Integration Test Suite: Live Providers, Tools & PostgreSQL
- [x] Provision PostgreSQL 17 with `pgvector` container via `docker-compose.yml`.
- [x] Implement and execute live integration test suite (13 tests total) against real Google Gemini models (`gemini-3.5-flash-lite`) and PostgreSQL:
  - `tests/integration/test_live_genai_core.py`: Live `LLMService` structure/text and `MemoryService` LTM/episodic operations.
  - `tests/integration/test_live_module17_agent.py`: Live `BoundedAgent` querying customer accounts, tickets, and pgvector knowledge base.
  - `tests/integration/test_live_planning_components.py`: Live `PlanGenerator` DAG decomposition, `StepVerifier` dual-gate model judge, and `ReplanningEngine` dynamic recovery.
  - `tests/integration/test_live_assistant_e2e.py`: Live end-to-end `PlanAndExecuteAssistant` multi-step run with final deliverable synthesis, human review rejection gating, and `RunBudget` deadline enforcement.

### Milestone 8: Pre-Push Quality Gate, Typing & Documentation
- [x] Verify `pre_push.sh` runs cleanly:
  - `ruff check .` with zero errors.
  - `ruff format --check .` with zero diffs.
  - `mypy src tests` with zero type errors (checked 84 source files).
  - `pytest` with 100% unit tests passing (121 unit tests passed, 4 mock integration tests passed).
- [x] Final verification and documentation updates for audit findings.
