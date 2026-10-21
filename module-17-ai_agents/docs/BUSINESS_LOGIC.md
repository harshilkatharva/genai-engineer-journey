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
| `gemini-1.5-flash` / default | $0.075 | $0.30 |
| `gemini-1.5-pro` | $1.25 | $5.00 |
| `gpt-4o-mini` | $0.15 | $0.60 |
| `gpt-4o` | $2.50 | $10.00 |
| `claude-3-5-sonnet` | $3.00 | $15.00 |

If usage is not reported by the provider or in mock tests, costs are reported honestly as estimated or zero rather than fabricated.

---

## 5. Storage Defaults

- **Persistent Memory:** PostgreSQL via Docker (`DATABASE_CONNECTION_CONVERSATION_URL` or `postgres://postgres:postgres@127.0.0.1:5432/ai_search`).
- **Working Memory:** In-memory store provided by `genai-core` (`InMemoryMemoryStore`).
- **Approval Store:** `PostgresApprovalStore` storing requests and decisions in PostgreSQL, with `InMemoryApprovalStore` for unit tests.
- **Internal Knowledge Base:** `RetrievalService` with `InMemoryVectorStore` seeded with policy articles.
