# Module 17: Bounded Research and Support Agent

A production-grade, plain-Python bounded agent combining internal knowledge-base retrieval, lightweight web search, customer and ticket lookup, and controlled support and memory-management operations.

Built strictly on top of `packages/genai-core` as the reusable AI infrastructure foundation, without importing or depending on any legacy module source code.

---

## 1. Project Overview

The Bounded Research and Support Agent is designed to solve real-world enterprise customer support tasks while strictly enforcing execution boundaries, preventing runaway API spend, mitigating prompt injection attacks, and gating sensitive operations behind human approval with durable pause/resume capabilities.

### Key Capabilities

- **16 Classified Domain Tools:** Read-only (6), reversible write (3), and irreversible gated actions (7).
- **Bounded Execution Loop:** Enforces step limits, token budgets, USD cost limits, wall-clock deadlines, and repeated-call loop detection.
- **Durable Human-in-the-Loop (HITL):** Sensitive actions pause execution, persist pending approval state to PostgreSQL (`PostgresApprovalStore`), and resume seamlessly after an operator decision, even across process restarts.
- **Two-Tier Persistent Memory:** PostgreSQL-backed long-term memory with user isolation, working context enrichment, and episodic action logging with idempotency checks to prevent duplicate side effects.
- **Defense-in-Depth for Untrusted Data:** Knowledge-base and web content are sanitized and structurally wrapped in `<untrusted_content>` delimiters with instruction disclaimers to prevent indirect prompt injections.
- **Structured JSON Transcripts:** Comprehensive run telemetry recording per-step latency, token counts, pricing calculations, loop detections, and sensitive argument redactions.
- **15-Goal Benchmark Evaluation Suite:** Standardized evaluation pipeline measuring Success Rate, Mean Steps, and Mean Cost per goal.

---

## 2. Architecture & Dependency Rules

```
module-17-ai_agents/
├── src/research_and_support_agent/
│   ├── config.py             # Agent runtime configuration & budgets
│   ├── models.py             # Domain models (CustomerAccount, SupportTicket, Reversals)
│   ├── agent.py              # BoundedAgent orchestrator wrapping ToolLoopRunner
│   ├── tools/
│   │   ├── read_only.py      # 6 Read-only tools (search_kb, web_search, lookup, etc.)
│   │   ├── reversible.py     # 3 Reversible tools (save_memory, update_ticket, escalate)
│   │   ├── irreversible.py   # 7 Irreversible tools (refunds, deletions, purges)
│   │   ├── sanitizer.py      # Prompt injection detection & structural tag isolation
│   │   └── registry.py       # Factory registering all tools with GenAI-Core ToolRegistry
│   ├── hitl/
│   │   ├── postgres_store.py # PostgresApprovalStore implementing ApprovalStore protocol
│   │   └── authorization.py  # User-ownership & support-role authorization validator
│   ├── memory/
│   │   └── manager.py        # Pre-turn recall injection & episodic idempotency logging
│   ├── telemetry/
│   │   ├── pricing.py        # Token pricing rate cards (Gemini, OpenAI, Claude)
│   │   └── transcript.py     # Structured JSON run transcripts with PII redaction
│   └── eval/
│       ├── dataset.py        # 15 benchmark goal scenarios (EvalDataset)
│       ├── metrics.py        # Step, cost, and keyword compliance evaluators
│       └── runner.py         # Benchmark execution runner & summary generator
├── docs/                     # Project Change Control source of truth
│   ├── GOAL.md
│   ├── DECISIONS.md
│   ├── FLOW.md
│   └── BUSINESS_LOGIC.md
├── scripts/
│   ├── pre_push.sh           # Linting, formatting, type checking, and test runner
│   └── run_eval.py           # Runnable 15-goal evaluation benchmark script
├── tests/
│   ├── unit/                 # Domain tools, pricing, sanitizer, authorization, transcripts
│   └── integration/          # Boundaries, memory, HITL pause/resume, eval benchmark
├── docker-compose.yml        # PostgreSQL 16 + pgvector container
└── pyproject.toml            # Dependencies and tool configurations
```

### Strict Isolation Rule
All reusable AI capabilities (LLM integration, retrieval pipelines, memory protocols, tool registry, loop bounding, evaluation frameworks) are consumed strictly from `packages/genai-core`. No code is imported from `module-09-rag`, `module-16-memory`, or any other previous module.

---

## 3. Tool Classification & Risk Policy

Every tool is registered with explicit schemas, documentation, and risk levels:

| Tool Name | Risk Level | Description |
|---|---|---|
| `search_knowledge_base` | `READ_ONLY` | Semantic retrieval from internal documents via `RetrievalService`. |
| `search_web` | `READ_ONLY` | Queries DuckDuckGo HTML, extracts text, bounds length, and includes source URLs. |
| `get_longterm_memory` | `READ_ONLY` | Retrieves user-scoped facts and preferences from long-term memory. |
| `get_episodic_memory` | `READ_ONLY` | Retrieves past action history and outcomes for the user. |
| `ticket_status` | `READ_ONLY` | Looks up support ticket state, queue, and priority. |
| `lookup_customer_account` | `READ_ONLY` | Retrieves customer tier, status, and account balance. |
| `save_longterm_memory` | `REVERSIBLE_WRITE` | Stores a user preference/fact; reversible via deletion. |
| `update_ticket_status` | `REVERSIBLE_WRITE` | Updates ticket state while recording `previous_status` to enable rollback. |
| `escalate_to_tier2` | `REVERSIBLE_WRITE` | Escalates ticket to Tier-2 queue; reversible via cancellation. |
| `delete_longterm_memory_all` | `IRREVERSIBLE` | Clears all long-term memories for authorized user (requires user approval). |
| `delete_longterm_memory_specific` | `IRREVERSIBLE` | Deletes a specific memory record (requires user approval). |
| `delete_episodic_memory_all` | `IRREVERSIBLE` | Clears all episodic actions for authorized user (requires user approval). |
| `delete_episodic_memory_episodes` | `IRREVERSIBLE` | Deletes specific episodic records (requires user approval). |
| `purge_expired` | `IRREVERSIBLE` | Removes only expired records; never touches active memories (support approval). |
| `execute_refund` | `IRREVERSIBLE` | Credits funds to a customer; requires support approval and idempotency check. |
| `delete_customer_data` | `IRREVERSIBLE` | Purges customer personal data within authorized scope (support approval). |

---

## 4. Human-in-the-Loop (HITL) Workflow

When an irreversible tool is invoked:
1. `ToolExecutor` pauses execution and constructs an `ApprovalRequest` containing `request_id`, `tool_name`, `arguments`, and `risk_level`.
2. The request is persisted in PostgreSQL (`PostgresApprovalStore`) or in-memory fallback.
3. The agent cleanly returns `ToolLoopStatus.APPROVAL_REQUIRED` along with the serialized pending request.
4. An authorized human reviewer reviews the request and records an `ApprovalDecision`:
   - Memory deletions must be approved by the **memory owner** (`user_id`).
   - Refunds, purges, and customer deletions must be approved by **support team** (`role="support_team"`).
   - Reviewers may optionally submit modified arguments (e.g. adjusting refund amount).
5. Application code calls `agent.resume()`:
   - Validates that the LLM did not self-approve.
   - Revalidates reviewer authorization and argument schemas.
   - Executes the gated tool with `bypass_approval=True`.
   - Records the outcome to episodic memory.

---

## 5. Bounded Execution Safeguards

The agent enforces 5 strict runtime constraints:
- **Max Steps:** Default 10 steps. Exceeding triggers `ToolLoopStatus.MAX_ITERATIONS_REACHED`.
- **Token Budget:** Default 10,000 tokens. Exceeding triggers `ToolLoopStatus.BUDGET_EXCEEDED`.
- **Cost Budget:** Default $0.05 USD. Exceeding triggers `ToolLoopStatus.BUDGET_EXCEEDED`.
- **Wall-Clock Deadline:** Default 30.0s. Exceeding triggers `ToolLoopStatus.DEADLINE_EXCEEDED`.
- **Loop Detection:** Repeated identical calls trigger warnings or termination via `LoopDetector`.

---

## 6. Local Setup & Docker Instructions

### Prerequisites
- Python 3.12+
- `uv` package manager
- Docker and Docker Compose (for PostgreSQL + pgvector)

### 1. Launch PostgreSQL with Docker
```bash
docker compose up -d
```
This spins up PostgreSQL 16 on port `5432` with the `pgvector` extension, initializing `research_agent_db`.

### 2. Install Dependencies
```bash
uv sync
```

### 3. Run Quality Gates (Pre-push verification)
```bash
./scripts/pre_push.sh
```
Runs:
1. `ruff check .` (Linting)
2. `ruff format --check .` (Formatting)
3. `mypy src tests` (Type checking)
4. `pytest -v` (Full unit & integration test suite)

### 4. Run the 15-Goal Benchmark Evaluation
```bash
uv run python scripts/run_eval.py
```

Sample output:
```
================================================================================
MODULE 17: BOUNDED RESEARCH & SUPPORT AGENT BENCHMARK REPORT
================================================================================
Total Goals Evaluated: 15
Passed Samples:        15
--------------------------------------------------------------------------------
Key Performance Indicators (KPIs):
  • Success Rate:      100.0%
  • Mean Steps:        1.4 steps / goal
  • Mean Cost:         $0.000044 USD / goal
  • Total Run Cost:    $0.000660 USD
================================================================================
```

---

## 7. Structured JSON Transcripts

Each agent execution produces a structured transcript stored in `transcripts/run_<run_id>.json`:
```json
{
  "run_id": "run-a1b2c3d4",
  "goal": "Refund customer C-101 $50 for damaged packaging",
  "status": "COMPLETED",
  "start_time": "2026-10-21T14:40:00.000Z",
  "end_time": "2026-10-21T14:40:02.150Z",
  "total_latency_seconds": 2.15,
  "total_tokens": 320,
  "total_cost_usd": 0.000048,
  "termination_reason": "completed",
  "steps": [
    {
      "step_index": 1,
      "llm_model": "gemini-1.5-flash",
      "tool_calls": [
        {
          "tool_name": "lookup_customer_account",
          "arguments": {"customer_id": "C-101"}
        }
      ],
      "tool_results": [
        {
          "tool_name": "lookup_customer_account",
          "status": "success",
          "content": "Customer C-101: Tier Gold, Balance: $100.00"
        }
      ],
      "input_tokens": 120,
      "output_tokens": 40,
      "cost_usd": 0.000021,
      "latency_seconds": 0.45
    }
  ]
}
```
Sensitive arguments (e.g. credit card tokens, passwords, customer SSNs) are automatically redacted using configurable pattern masking.

---

## 8. Limitations & Future Work

- **Live Web Scraping:** `search_web` relies on DuckDuckGo HTML scraping. Rate limiting may occur under heavy concurrent loads; a dedicated proxy or search API (e.g. Tavily/SerpAPI) should be configured in high-traffic production environments.
- **Idempotency Window:** The episodic idempotency check currently verifies duplicate operations within the user's active session. Cross-session long-term idempotency should use durable unique transaction idempotency keys in PostgreSQL.
