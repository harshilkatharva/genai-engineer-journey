# Module 17: Bounded Research and Support Agent — Goals & Progress

## 1. Project Goal
Build a plain-Python, bounded research and support agent that combines internal knowledge-base retrieval (`RetrievalService`), lightweight web search, customer account and ticket lookups, and controlled support and memory-management actions. The agent must enforce strict execution boundaries (steps, tokens, cost, deadline, loop detection), use persistent memory (PostgreSQL for long-term and episodic memory, in-memory for working context), require human-in-the-loop approval for irreversible actions with durable pause/resume, emit structured JSON transcripts, and validate against a 15-goal benchmark evaluation suite.

## 2. Source of Truth
- `GOAL.md` — Project milestones, completed items, and pending tasks.
- `DECISIONS.md` — Append-only architectural decision log.
- `FLOW.md` — Component interactions and end-to-end data/execution flow.
- `BUSINESS_LOGIC.md` — Confirmed business rules, tool risks, limits, and defaults.

## 3. Scope & Milestones

### Milestone 1: Environment & Tool Infrastructure
- [x] Configure `module-17-ai_agents/pyproject.toml` with `packages/genai-core` local path dependency.
- [x] Implement 6 Read-Only Tools:
  - `search_knowledge_base` (RAG via `RetrievalService`)
  - `search_web` (Lightweight DuckDuckGo search + text extraction)
  - `get_longterm_memory` (User-isolated long-term recall)
  - `get_episodic_memory` (User-isolated past actions)
  - `ticket_status` (Support ticket state retrieval)
  - `lookup_customer_account` (Customer profile & tier lookup)
- [x] Implement 3 Reversible Tools:
  - `save_longterm_memory` (Store user fact/preference)
  - `update_ticket_status` (State transition with undo tracking)
  - `escalate_to_tier2` (Escalation with reversal mechanism)
- [x] Implement 7 Irreversible Gated Tools:
  - `delete_longterm_memory_all` (Delete all long-term records for user)
  - `delete_longterm_memory_specific` (Delete selected record ID)
  - `delete_episodic_memory_all` (Delete all episodic records for user)
  - `delete_episodic_memory_episodes` (Delete selected episode records)
  - `purge_expired` (Purge only expired records)
  - `execute_refund` (Financial refund with idempotency check)
  - `delete_customer_data` (Authorized customer data deletion)
- [x] Implement `ToolOutputSanitizer` defending against prompt injection and jailbreaks.

### Milestone 2: Human-in-the-Loop (HITL) & Storage Persistence
- [x] Implement `PostgresApprovalStore` implementing `ApprovalStore` protocol from `genai-core`.
- [x] Implement trusted authorization policy binding user/role, action, target records, and exact arguments.
- [x] Support pause, database persistence, operator decision (approve, reject, modify args), and resumption.
- [x] Configure `PostgresMemoryStore` for persistent memory with seed data and Docker setup.

### Milestone 3: Bounded Agent Execution & Telemetry
- [x] Build `BoundedAgent` wrapping `ToolLoopRunner`, `ToolBudget`, and `LoopDetector`.
- [x] Enforce and test: max steps, cumulative tokens, monetary cost budget, wall-clock deadline, loop detection.
- [x] Implement model pricing calculator mapping actual token usage to USD.
- [x] Implement `TranscriptRecorder` serializing `AgentTranscript` JSON logs.

### Milestone 4: Evaluation Benchmark & Quality Gates
- [x] Curate 15 benchmark goal scenarios in `src/research_and_support_agent/eval/dataset.py`.
- [x] Implement runnable benchmark runner (`scripts/run_eval.py`) reporting:
  - Success Rate: 100%
  - Mean Steps: 1.4 steps / goal
  - Mean Cost: $0.000044 USD / goal
- [x] Add unit and integration tests covering all tools, limits, memory, and approval workflows (28/28 tests passing).
- [x] Run `ruff check`, `ruff format`, `mypy src tests`, and `pytest` via `scripts/pre_push.sh` (100% passing).

