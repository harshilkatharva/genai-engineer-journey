# Module 17: Bounded Research and Support Agent — Confirmed Business Logic

## 1. Tool Classification & Risk Policy

The agent exposes exactly 16 tools categorized into three operational risk tiers:

### Read-Only Tools (No state mutation)
1. `search_knowledge_base` (`RiskLevel.READ_ONLY`): Retrieves documentation chunks using `RetrievalService`.
2. `search_web` (`RiskLevel.READ_ONLY`): Queries public web pages via DuckDuckGo, extracts text with length bounding, and tags sources.
3. `get_longterm_memory` (`RiskLevel.READ_ONLY`): Retrieves stored preferences/facts belonging to the authorized `user_id`.
4. `get_episodic_memory` (`RiskLevel.READ_ONLY`): Retrieves past action history and outcomes for the authorized `user_id`.
5. `ticket_status` (`RiskLevel.READ_ONLY`): Looks up current support ticket details by `ticket_id`.
6. `lookup_customer_account` (`RiskLevel.READ_ONLY`): Looks up customer profile, balance, and tier by `customer_id`.

### Reversible Tools (State modified with explicit undo mechanism)
7. `save_longterm_memory` (`RiskLevel.REVERSIBLE_WRITE`): Stores durable memory via `MemoryService.remember()`. Reversible by deleting the created record.
8. `update_ticket_status` (`RiskLevel.REVERSIBLE_WRITE`): Changes ticket state while recording `previous_status` to allow status rollback.
9. `escalate_to_tier2` (`RiskLevel.REVERSIBLE_WRITE`): Escalates ticket to Tier-2 with escalation tracking; reversible by canceling escalation.

### Irreversible Tools (Gated by Human Approval)
10. `delete_longterm_memory_all` (`RiskLevel.IRREVERSIBLE`): Clears all long-term memories for the authorized user.
11. `delete_longterm_memory_specific` (`RiskLevel.IRREVERSIBLE`): Deletes a specified long-term memory record by `record_id`.
12. `delete_episodic_memory_all` (`RiskLevel.IRREVERSIBLE`): Clears all episodic action logs for the authorized user.
13. `delete_episodic_memory_episodes` (`RiskLevel.IRREVERSIBLE`): Deletes specific episodic records by ID.
14. `purge_expired` (`RiskLevel.IRREVERSIBLE`): Purges only expired long-term memory records. Never purges active records.
15. `execute_refund` (`RiskLevel.IRREVERSIBLE`): Credits funds to customer. Gated by support-team approval and guarded by episodic idempotency.
16. `delete_customer_data` (`RiskLevel.IRREVERSIBLE`): Purges customer personal data within authorized scope. Gated by support-team approval.

---

## 2. Human-in-the-Loop (HITL) Authorization Policy

- **Memory Deletion Policy:** Operations `delete_longterm_memory_all`, `delete_longterm_memory_specific`, `delete_episodic_memory_all`, and `delete_episodic_memory_episodes` MUST be confirmed by the specific user who owns the memory.
- **Support-Team Action Policy:** Operations `execute_refund`, `purge_expired`, and `delete_customer_data` MUST be approved by an authorized support-team operator.
- **Self-Approval Prohibition:** The LLM cannot approve its own actions or masquerade as an authorized reviewer.
- **Revalidation on Resume:** Resumed operations must re-verify that the acting reviewer had appropriate permissions, and modified arguments must be validated against the tool's Pydantic schema before execution.

---

## 3. Bounded Execution Limits & Defaults

| Limit Parameter | Default Value | Behavior on Exhaustion |
|---|---|---|
| `max_steps` | 10 iterations | Halts loop with `ToolLoopStatus.MAX_ITERATIONS_REACHED` |
| `max_tokens` | 10,000 tokens | Halts loop with `ToolLoopStatus.BUDGET_EXCEEDED` |
| `max_cost_usd` | $0.05 USD | Halts loop with `ToolLoopStatus.BUDGET_EXCEEDED` |
| `deadline_monotonic` | `time.monotonic() + 30.0` | Halts loop with `ToolLoopStatus.DEADLINE_EXCEEDED` |
| `max_repeated_calls` | 3 repeats | `LoopDetector` injects self-correction prompt into context |

---

## 4. Model Pricing Rate Card

Pricing is calculated per completed LLM step based on model ID:

| Model Family | Prompt Tokens (per 1M) | Completion Tokens (per 1M) |
|---|---|---|
| `gemini-3.5-flash-lite` (Default) | $0.075 | $0.30 |
| `gemini-1.5-flash` | $0.075 | $0.30 |
| `gemini-1.5-pro` | $1.25 | $5.00 |
| `gpt-4o-mini` | $0.15 | $0.60 |
| `gpt-4o` | $2.50 | $10.00 |
| `claude-3-5-sonnet` | $3.00 | $15.00 |

If usage is not reported by the provider or in mock tests, costs are reported honestly as estimated or zero rather than fabricated.

---

## 5. Storage Defaults & Schemas

- **Persistent Memory Store:** PostgreSQL tables `memory_lt_records` (long-term facts with 384-dimensional pgvector embeddings) and `memory_ep_episodes` (episodic action logs with idempotency keys).
- **Relational Domain Store:** PostgreSQL tables `support_customers` (accounts, tiers, balances, risk flags) and `support_tickets` (tickets, statuses, priority, history).
- **Knowledge-Base Vector Store:** PostgreSQL table `document_chunks` (384-dimensional embeddings of policy docs via `PGVectorStoreAdapter`).
- **Approval Store:** `PostgresApprovalStore` (`agent_approval_requests` table) storing HITL requests, statuses, and decisions in PostgreSQL, with `InMemoryApprovalStore` for isolated unit tests.
- **Working Memory:** In-memory store provided by `genai-core` (`InMemoryMemoryStore`).

---

## 6. Evaluation Execution Modes

When running `scripts/run_eval.py` interactively, two options are presented:
1. `1. deterministic_mock eval`: Offline deterministic mock mode, suitable for fast CI/CD validation without API costs or external network dependencies.
2. `2. realistic eval run`: Live mode connecting to local PostgreSQL, executing real Google Gemini 3.5 Flash Lite API calls, DuckDuckGo web search queries, vector similarity lookups, and HITL authorization flows.

---

## 7. Test Execution & Integration Gating Policy

- **Unit Tests:** Fast, isolated unit tests run by default with zero external dependencies (`pytest tests/unit/` or zero-argument `pytest`).
- **Integration Tests:** Integration tests under `tests/integration/` or marked `@pytest.mark.integration` execute ONLY when `INTEGRATION_TEST=1` (or `true`, `yes`, `on`) is present in the environment or `.env` file.
- **Pre-Push Quality Gate:** Running `./scripts/pre_push.sh` verifies formatting, typing, linting, and runs unit tests. When `INTEGRATION_TEST=1` is exported, integration tests are verified in the quality gate as well.

