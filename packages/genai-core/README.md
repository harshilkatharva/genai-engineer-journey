# genai-core

Production-grade, modular, and resilient Python core package providing unified GenAI capabilities across `genai-engineer-journey` modules and downstream applications.

`genai-core` standardizes model access, embeddings, vector retrieval, rate limiting, retries, and privacy-safe observability behind clean, typed, and interchangeable public interfaces.

---

## Capabilities

### 1. Unified LLM Subsystem (`LLMService`)
- **Protocol-driven multi-provider abstraction**: Unified interface (`LLMProvider`) supporting:
  - `GoogleProvider`: Native Gemini models via `google-genai` SDK.
  - `OpenAIProvider`: OpenAI Chat & Structured Outputs via `openai` SDK.
  - `AnthropicProvider`: Claude Messages API via `anthropic` SDK.
- **First-class capabilities**:
  - `complete(prompt)`: Text completion with system instructions and temperature controls.
  - `complete_structure(prompt, response_schema)`: Schema-enforced structured output (Pydantic) with automated validation retries.
  - `complete_tool(messages, tools, tool_choice)`: Function declarations and multi-turn tool calling.
  - `stream(messages)`: Asynchronous streaming yielding normalized token deltas.
  - `complete_all(prompts)`: Concurrent, error-shielded batch completions.
- **Normalized responses**: `LLMResponseModel` unifying text, `structured_response`, `tool_calls`, token usage, model version, and latency.

### 2. Unified Embedding Subsystem (`EmbeddingService`)
- **Interchangeable providers**:
  - `LocalEmbeddingProvider`: Fast local embeddings via `sentence-transformers` (e.g. `all-MiniLM-L6-v2`, optional extra `[local]`).
  - `OpenAIEmbeddingProvider`: Hosted embeddings via `text-embedding-3-small` with configurable output dimensions and usage tracking.
  - `GoogleEmbeddingProvider`: Hosted embeddings via `text-embedding-005` with L2 normalization and `output_dimensionality` control.
- **Production reliability**:
  - Strict initialization dimension validation (rejects unsupported configurations fast).
  - Internal batch slicing with guaranteed 1-to-1 input sequence preservation.
  - Non-blocking async execution (`asyncio.to_thread` for local PyTorch inference).
  - Honest token usage accounting (never estimates or fabricates token usage).

### 3. Unified Retrieval Subsystem (`RetrievalService`)
- **Vector store adapters (`VectorStoreAdapter`)**:
  - `InMemoryVectorStore`: Zero-dependency, pure-Python vector store with exact cosine similarity, metadata dictionary filtering, and score thresholding.
  - `PGVectorStoreAdapter`: PostgreSQL + pgvector adapter with dependency-injected connection pooling (`psycopg_pool`), JSONB metadata containment filtering (`@>`), parameterized SQL, and identifier whitelisting.
- **Retrieval workflows**:
  - Dense semantic search with `top_k`, `score_threshold`, and metadata filtering.
  - Seamless chunk ingestion (`upsert`, `delete`, `index_documents`).
  - Automated vectorization of raw document chunks via attached `EmbeddingService`.
  - Strict dimension compatibility verification between embeddings and vector stores.

### 4. Shared Resilience Subsystem (`genai_core.resilience`)
- **Async Rate Limiter (`RateLimiter`)**:
  - Dual in-memory token buckets enforcing Requests Per Minute (RPM) and Tokens Per Minute (TPM).
  - Dynamic pause handling when upstream services return HTTP `Retry-After` headers.
- **Retry Policy (`RetryPolicy` & `execute_with_retry`)**:
  - Exponential backoff with randomized jitter and per-attempt timeouts.
  - Bounded overall deadlines across multi-batch and chained operations.
  - Intelligent classification: retries transient errors (HTTP 408/429/5xx, connection resets), fails fast on permanent errors (400, 401, 403, quota exhaustion, validation failures).

### 5. Caller-Controlled Observability (`genai_core.observability`)
- **Zero side-effects by default**: Configured with `logging.NullHandler()`; never creates log directories or files without explicit caller opt-in.
- **Automated redaction**: Sanitizes API keys (Google, OpenAI, Anthropic), Bearer headers, and sensitive fields before writing logs.
- **Context tracking**: Async-safe request correlation (`set_request_id`, `observability_context`).
- **Fault-tolerant dispatch**: `safe_log` prevents logging errors from interrupting primary GenAI workflows.

---

## Installation

### Base Installation
Installs core contracts, LLM service, in-memory vector store, and resilience:
```bash
uv pip install -e packages/genai-core
```

### Optional Extras
- **Local Embeddings** (includes `sentence-transformers` and PyTorch):
  ```bash
  uv pip install -e "packages/genai-core[local]"
  ```
- **PostgreSQL / pgvector Adapter** (includes `psycopg` and `pgvector`):
  ```bash
  uv pip install -e "packages/genai-core[postgres]"
  ```
- **All Extras**:
  ```bash
  uv pip install -e "packages/genai-core[local,postgres]"
  ```

---

## Configuration

Settings are managed via `Settings` and can be configured through `.env` or environment variables:

| Environment Variable | Description | Default |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | Google Gemini API key | `""` |
| `OPENAI_API_KEY` | OpenAI API key | `""` |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | `""` |
| `MAX_STRUCTURED_ERROR_RETRY` | Default retries on JSON schema validation errors | `2` |
| `INTEGRATION_TEST` | Enable live API integration test execution | `0` |
| `LOG_LEVEL` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `"INFO"` |
| `LOG_FILE` | Optional path to write rotating log files | `""` |

---

## Quickstart

### 1. LLM Service
```python
import asyncio
from pydantic import BaseModel
from genai_core import LLMService, GoogleProvider


class EntityExtraction(BaseModel):
    entities: list[str]
    sentiment: str


async def main():
    service = LLMService(default_provider=GoogleProvider(model="gemini-2.5-flash"))

    # Basic completion
    response = await service.complete("Explain vector embeddings in one sentence.")
    print(response.content)

    # Structured output
    structured = await service.complete_structure(
        prompt="Apple announced new MacBooks in Cupertino. Great feedback so far!",
        response_schema=EntityExtraction,
    )
    print(structured.structured_response)


asyncio.run(main())
```

### 2. Embedding Service
```python
import asyncio
from genai_core import EmbeddingService, LocalEmbeddingProvider


async def main():
    service = EmbeddingService(
        provider=LocalEmbeddingProvider(model="all-MiniLM-L6-v2"),
        dimensions=384,
        batch_size=32,
    )

    response = await service.embed(["Hello world", "Semantic search with genai-core"])
    print(
        f"Generated {len(response.embeddings)} vectors of dimension {len(response.embeddings[0])}"
    )


asyncio.run(main())
```

### 3. Retrieval Service
```python
import asyncio
from genai_core import (
    DocumentChunk,
    EmbeddingService,
    InMemoryVectorStore,
    LocalEmbeddingProvider,
    RetrievalQuery,
    RetrievalService,
)


async def main():
    embedding_service = EmbeddingService(
        provider=LocalEmbeddingProvider(),
        dimensions=384,
    )
    store = InMemoryVectorStore(dimensions=384)
    retriever = RetrievalService(embedding_service=embedding_service, store=store)

    # Index document chunks
    chunks = [
        DocumentChunk(id="c1", text="genai-core provides unified LLM and embedding access."),
        DocumentChunk(id="c2", text="PostgreSQL pgvector enables persistent similarity search."),
    ]
    await retriever.index_documents(chunks)

    # Dense semantic search
    query = RetrievalQuery(text="How does genai-core handle embeddings?", top_k=1)
    response = await retriever.search(query)
    for result in response.results:
        print(f"Score: {result.score:.4f} | Content: {result.chunk.text}")


asyncio.run(main())
```

---

## Project Structure

```text
packages/genai-core/
├── docs/                     # Architectural single sources of truth
│   ├── GOAL.md               # Purpose, milestones, and progress
│   ├── FLOW.md               # End-to-end component and package flows
│   ├── DECISIONS.md          # Append-only architectural decisions
│   └── BUSINESS_LOGIC.md     # Confirmed contracts and validation rules
├── src/genai_core/
│   ├── core/                 # Settings and package configuration
│   ├── embedding/            # Local, OpenAI, and Google embedding providers
│   ├── exceptions/           # Strongly-typed exceptions
│   ├── llm/                  # LLM provider adapters and protocol
│   ├── models/               # Pydantic data models (LLM, embedding, retrieval)
│   ├── observability/        # Logging, redaction, formatting, and tracing
│   ├── resilience/           # Rate limiting and retry policies
│   ├── retrieval/            # Vector store protocol, InMemory and PGVector stores
│   └── services/             # LLMService, EmbeddingService, RetrievalService
├── tests/
│   ├── unit/                 # Fast, isolated unit tests (no live network)
│   │   ├── embedding/
│   │   ├── llm/
│   │   ├── observability/
│   │   └── retrieval/
│   └── integration/          # Live API integration tests (gated by INTEGRATION_TEST=1)
│       ├── llm/
│       └── observability/
├── pyproject.toml
└── README.md
```

---

## Running Tests

### Unit Tests
```bash
uv run pytest packages/genai-core/tests/unit
```

### Live Integration Tests
Live tests require provider API keys and `INTEGRATION_TEST=1`:
```bash
INTEGRATION_TEST=1 uv run pytest packages/genai-core/tests/integration
```

### Pre-Push Quality Gate
Runs `ruff` rule checks, `ruff` formatting, and `mypy` strict type checking:
```bash
bash packages/genai-core/scripts/pre_push.sh
```
