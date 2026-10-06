# Multi-Layer Memory System

A standalone Python memory package for conversational applications. It provides
working, long-term, and episodic memory services without depending on an agent
framework. User-facing memory management is available as Python service methods;
the package does not define HTTP routes.

## Architecture

- **Working memory** holds one user's active conversation context in a process-local
  registry. `WorkingMemory.snapshot(token_budget)` preserves every pinned goal
  first, then retains the newest contiguous suffix of messages that fits. If pinned
  goals alone exceed the budget, they remain present and the result reports the
  actual token count (which can exceed the requested budget).
- **Long-term memory** extracts durable memories through the shared `LLMService`,
  embeds them through an injectable interface (Sentence Transformers by
  default), and stores vectors in PostgreSQL/pgvector. Retrieval combines cosine
  similarity with a recency score and demotes explicitly superseded memories.
- **Episodic memory** stores structured attempted/completed/failed actions and
  exposes `check()` to recommend whether an action should run again. Checks are
  scoped to a user and task, and compare canonicalized action, target, and parameters.
- `MemoryService` provides view, delete-one, and delete-all operations. Delete-all
  removes long-term and episodic rows in one database transaction and clears the
  user's working-memory entries in the current process.

`PostgresMemoryStore` is the PostgreSQL implementation. The services accept
protocol-compatible stores, extractors, and embedders to support tests and provider
changes without coupling memory policy to agents.

The package follows the separation used in Module 08's vector-search project:
embedding generation, retrieval orchestration, and database operations each have
their own package rather than being bundled into a memory module:

```text
src/memory_system/
├── core/       # shared environment-backed settings
├── db/         # PostgreSQL repository, pgvector queries, and schema
├── embedding/  # embedding provider abstraction and local model adapter
├── memory/     # working, long-term, episodic, extraction, and user APIs
├── models/     # typed memory and existing LLM schemas
└── retrieval/  # query embedding + semantic/recency retrieval orchestration
```

## Setup

Requires Python 3.12+ and PostgreSQL with the pgvector extension available to the
database user. Install this package and development dependencies using the
repository's package workflow (for example, `uv sync --dev`), then set the
environment variables shown in `.env.example`. By default, the memory store uses
`DATABASE_CONNECTION_CONVERSATION_URL`; tests prefer the optional
`MEMORY_TEST_DATABASE_URL` override and otherwise use the conversation database
URL. No credentials are checked into this project.

Apply the schema before connecting:

```sh
psql "$DATABASE_CONNECTION_CONVERSATION_URL" -v ON_ERROR_STOP=1 -f src/memory_system/db/schema.sql
```

The migration defines `vector(384)`. Keep `MEMORY_EMBEDDING_DIMENSION=384` and use a
384-dimensional embedding model, or adapt the migration to the dimension of a
different model before deployment. The default Sentence Transformers model is
`sentence-transformers/all-MiniLM-L6-v2`.

Create and close the async store at application startup/shutdown:

```python
from memory_system import PostgresMemoryStore
from memory_system.core import get_settings

store = await PostgresMemoryStore.connect(get_settings().database_url)
try:
    # Construct and use memory services with this store.
    ...
finally:
    await store.close()
```

Construct `LLMMemoryExtractor` with the shared `LLMService` (which implements the
`LLMServices` completion interface) and pass it to `LongTermMemory` to enable
conversation extraction. The extractor sends typed `LLMManagerRequest` values
through `LLMService.complete()` and converts provider errors into explicit
`MemoryExtractionError`s. Configure provider API keys through the existing
environment-backed settings.

## Python APIs

```python
from memory_system import (
    ConversationMessage,
    EpisodicMemory,
    LLMMemoryExtractor,
    LLMService,
    LongTermMemory,
    MemoryCandidate,
    PostgresMemoryStore,
    WorkingMemoryRegistry,
)
from memory_system.services.memory_service import MemoryService

working = WorkingMemoryRegistry()
context = working.get("user-123", "conversation-456")
context.pin_goal("Plan a three-day vegetarian menu")
context.add_message(ConversationMessage(role="user", content="Avoid peanuts."))
trimmed = context.snapshot(token_budget=1_000)

llm_service = LLMService()
long_term = LongTermMemory(store, extractor=LLMMemoryExtractor(llm_service))
await long_term.remember_conversation(
    "user-123",
    [ConversationMessage(role="user", content="I prefer vegetarian meals.")],
)
results = await long_term.retrieve("user-123", "dietary preferences")
saved = await long_term.store(
    "user-123", MemoryCandidate(content="Prefers vegetarian meals", category="preference")
)

episodic = EpisodicMemory(store)
check = await episodic.check(
    "user-123", "menu-task-1", "send_email", "menu@example.test", {"subject": "Menu"}
)
if check.recommendation == "execute":
    # Execute the action, then record that it was attempted/completed.
    await episodic.record(
        "user-123", "menu-task-1", "send_email", "menu@example.test",
        {"subject": "Menu"}, status="completed",
    )

service = MemoryService(long_term, episodic, working)
visible_memories = await service.view_memories("user-123")
await service.delete_memory("user-123", saved.id)
await service.delete_all_memories("user-123")
# A scheduled maintenance job can physically purge records past their expiry.
await service.purge_expired()
```

`purge_expired()` returns the physical long-term and episodic row counts removed.
Schedule it using the deployment's maintenance runner. `check()` is a preflight
recommendation, not an atomic distributed lock. If
multiple workers can execute the same action concurrently, the calling application
must coordinate the check-and-execute sequence with its own idempotency mechanism.
The memory package intentionally does not execute actions.

## Retention and privacy

See [docs/retention-policy.md](docs/retention-policy.md) for retention defaults,
expiration/deletion behavior, user controls, and data that must not be stored as
long-term memory.

## Tests

Run unit tests and lint checks:

```sh
uv run pytest tests/unit
uv run ruff check src tests
```

PostgreSQL integration tests use `MEMORY_TEST_DATABASE_URL` when set, otherwise
`DATABASE_CONNECTION_CONVERSATION_URL` from `.env`. They create the idempotent
schema automatically and use a generated test user ID; prefer a disposable database.
Integration tests are disabled by default and run only when `INTEGRATION_TEST=1`
is set in `.env`.

```sh
INTEGRATION_TEST=1 uv run pytest tests/integration -m integration
```

Or set `INTEGRATION_TEST=1` in `.env` and run:

```sh
uv run pytest tests/integration -m integration
```