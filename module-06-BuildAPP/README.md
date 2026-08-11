# Module 6 — Multi-Feature AI Application

## Overview

This project extends the Module 5 single-endpoint AI service into a coherent, multi-feature AI application. It provides three AI-powered capabilities — **chat with conversation history**, **document summarization**, and **sentiment/intent classification** — built on a shared `core/` infrastructure layer. The application is designed as a foundation for future modules: Module 9 will add RAG on top of it, and Module 17 will add agent behavior, so the architecture prioritizes clean separation of concerns over short-term simplicity.

The system is a FastAPI service backed by PostgreSQL, with server-managed conversation history, token-budget-aware truncation, per-request cost tracking, model fallback on provider failure, and centralized configuration via environment variables.

---

## Architecture

The codebase follows a **core + features** layout:

```
src/ai_app/
├── api/            # FastAPI app, routers, auth, rate limiting
├── core/           # Shared infrastructure used by every feature
├── db/             # Database access layer + schema
├── features/       # One folder per AI capability
├── models/         # Pydantic request/response/domain models
├── services/       # LLM provider abstraction 
└── utils/          # Logging, cost charts, helpers
```

**Why this structure:** every feature (chat, summarization, sentiment) needs the same things — conversation history, cost tracking, config, and an LLM client. Rather than duplicating that logic per feature, it lives once in `core/` and `services/`, and each feature in `features/<name>/service.py` composes those shared pieces around its own prompt and business logic. New features (RAG in Module 9, agents in Module 17) plug into this same pattern without touching existing code.

### Core Infrastructure (`core/`)

- **`AIConfig.py`** — A `pydantic-settings`-based configuration object controlling the default model/provider, fallback model/provider, temperature, conversation history token budget, and per-feature flags (e.g. `enable_rag`, `enable_agents`). Because it's environment-driven, swapping the active model is a config change, not a code change.
- **`config.py`** — Loads and validates required environment variables (API keys for Google/OpenAI/Anthropic, the service's own API key, database connection string) at startup, failing fast with a clear `ConfigError` if anything is missing.
- **`conversation_manager.py`** — Owns conversation lifecycle: starting conversations, persisting turns, retrieving history, and truncating that history to fit within the configured token budget (keeping the most recent turns and dropping the oldest first). It also implements a **response cache lookup** for the summarization feature (see below).
- **`cost_tracker.py`** — Converts input/output token counts into an estimated dollar cost using a per-model cost chart, so every AI call has a cost attached to it.

### Data Layer (`db/`)

Two tables back the whole system:

- **`conversations`** — one row per conversation, tied to a `user_id`.
- **`history`** — one row per message (system/user/assistant/tool), tagged with the `feature` that produced it (`chat`, `summarization`, `sentiment`), the model used, input/output token counts, estimated cost, and latency.

This schema is intentionally denormalized around `history` because it's the table every reporting and caching query runs against — usage-by-user, usage-by-feature, and cache lookups are all `WHERE`/`GROUP BY` queries on this one table.

### Features (`features/`)

Each feature — `chat`, `summarization`, `sentiment` — is a self-contained module with its own `service.py` and its own prompt template under `prompts/`. A feature's `service.py` is responsible for: pulling relevant conversation history, building the prompt, calling the LLM client, and (in the background, via FastAPI `BackgroundTasks`) persisting both the user message and the assistant response with their cost/token data.

Using `BackgroundTasks` for persistence means the user gets their response immediately, and the database write for cost tracking/history happens without adding to response latency.

### API Layer (`api/`)

FastAPI routers expose each feature under its own prefix (`/chat`, `/summarization`, `/sentiment`, `/start`, `/report`), all protected by an `X-API-KEY` header check and per-route rate limiting (10 requests/minute). A centralized exception handler normalizes error responses across all routes.

---

## Key Capabilities

### 1. Server-Managed Conversation History with Truncation

Conversation history is stored server-side (not client-side), and `ConversationManager._truncate_history` walks the history newest-first, accumulating tokens until it would exceed `AIConfig.conversation_history_max_token_size`, then stops. This guarantees every prompt sent to the LLM stays under the model's context budget regardless of how long a conversation gets, while always preserving the most recent (most relevant) turns.

### 2. Cost Tracking and Reporting

Every LLM call's input and output tokens are converted to an estimated cost via `CostTracker`, using a per-model price table. Costs are persisted per message in the `history` table. Two reporting endpoints aggregate this data:

- `GET /report/user?user_id=...` — total messages, tokens, and cost broken down by feature, for a given user.
- `GET /report/feature?feature=...` — total messages, tokens, and cost broken down by user, for a given feature.

This satisfies the "total cost per feature this week"-style aggregation requirement, and gives a foundation for usage-based billing or quota enforcement later.

### 3. Centralized, Environment-Driven Configuration

`AIConfig` (via `pydantic-settings`) centralizes every tunable: default model, fallback model, temperature, streaming toggle, history token budget, and feature flags. Because these are all backed by environment variables with defaults, changing the active model or toggling a feature is a deployment-time change, requiring no code edits or redeploys of logic.

### 4. Semantic Response Cache (Summarization)

Before calling the LLM for a summarization request, the system checks whether a sufficiently similar request was already served recently. The current implementation is **exact-match caching**: the incoming text is normalized (lowercased, whitespace-collapsed, trimmed) and compared against normalized user messages from the last 24 hours in the `history` table; if a match is found, the previously generated assistant response is returned directly, at zero additional token cost.

**Extending to embedding-based caching (Module 7):** once embeddings are available, the plan is to replace the exact-string match with a similarity search — embed the incoming (normalized) request, compare it against embeddings of recent cached requests (e.g., via cosine similarity in a vector column or a vector index), and treat anything above a similarity threshold as a cache hit. This would catch near-duplicate phrasings ("summarize this doc" vs. "please summarize this document") that exact-match currently misses, at the cost of needing an embedding call and a similarity search per request instead of a plain equality lookup.

### 5. Partial-Failure Handling

For flows with multiple stages (e.g., retrieval succeeds but generation fails), the system follows a deliberate degrade-gracefully strategy: rather than surfacing a hard error, it returns a partial result — the successfully retrieved information plus a clear message that generation failed — so the user isn't left with nothing when only one stage of a multi-step pipeline breaks.

---

## Testing Strategy

- **Unit tests** (`tests/unit/`) cover `ConversationManager` in isolation — starting conversations, adding history, retrieving and truncating history, and the edge case where a single newest message already exceeds the token budget (correctly truncating to an empty result rather than erroring).
- **Integration tests** (`tests/integration/`) exercise a full feature end-to-end — request → conversation management → real LLM call → cost tracking → persisted response — with the option to run against a live database and LLM provider (gated behind a `RUN_INTEGRATION_TESTS` environment flag so it doesn't run by default in CI).
- Database and LLM dependencies are mocked in unit tests (via `AsyncMock`) so the truncation and persistence logic can be verified without a real database connection.

---

## Smaller Exercises (Supplementary)

Alongside the mini-project, a set of standalone hands-on and coding exercises (in `excercise/`) were used to prototype and test individual concepts before integrating them into the main application:

- A standalone `ConversationManager` + `Message` dataclass, validating the truncation algorithm against a token budget before it was ported into the DB-backed version.
- `TrackUsages` / `UsageRecord` — a simpler, in-memory version of cost tracking and usage-record generation, tested with mocked LLM responses to confirm cost calculation and record fields end-to-end.
- `complete_with_fallback` — the fallback pattern in isolation, tested for both the "primary succeeds" and "primary fails, fallback succeeds" cases.
- `AIConfig` prototyped with `pydantic-settings`, confirming environment-variable-driven defaults work before wiring it into the real app.
- A partial-failure simulation (`question_answer`) modeling a retrieval-succeeds/generation-fails scenario and its documented recovery strategy.

These exercises de-risked the core algorithms (truncation, fallback, cost math) in isolation before they were embedded into the full FastAPI application described above.

---

## Summary

This project takes the single-feature Module 5 service and generalizes it into a multi-feature platform with shared, reusable infrastructure: environment-driven configuration, budget-aware conversation history, per-request cost tracking with aggregate reporting, provider fallback, and a first pass at response caching. The core/features split is deliberate — it's the seam future modules (RAG, agents) are expected to extend along, rather than a structure built just to pass this assignment.