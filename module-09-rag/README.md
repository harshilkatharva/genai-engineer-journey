# Module 09 — RAG Engine

A production-style, multi-tenant **Retrieval-Augmented Generation (RAG)** service built with FastAPI and Postgres/pgvector. This module is part of the [genai-engineer-journey](https://github.com/harshilkatharva/genai-engineer-journey) learning path and implements a full RAG pipeline — chunking, embedding, hybrid retrieval, reranking, prompt versioning, multi-provider LLM completion, and evaluation — as a set of small, independently swappable components rather than a single monolithic script.

## Overview

The service exposes a REST API that lets a client:

1. **Index** raw documents (chunk → embed → store) for a given tenant.
2. **Retrieve** the most relevant chunks for a query using vector, keyword, or hybrid search, optionally reranked with a cross-encoder.
3. **Chat** — run the full RAG loop (query rewriting → retrieval → prompt assembly → LLM completion) and get back a grounded answer.
4. **Evaluate** — generate a performance report from logged query traces for the currently running app version.

Every stage is implemented behind a small interface (a `Protocol` or a strategy dict) so that chunking strategies, retrieval strategies, query techniques, prompt versions, and LLM providers can all be swapped via configuration without touching calling code.

## Architecture

```
Query
  │
  ▼
┌─────────────────┐   query_expansion / query_HyDE (LLM-based)
│  Query Manager   │──▶ produces one or more search queries
└─────────────────┘
  │
  ▼
┌─────────────────┐   vector_search / keyword_search / hybrid_search
│ Retriever Manager│──▶ candidate chunks (top-k, score-normalized & merged)
└─────────────────┘
  │
  ▼
┌─────────────────┐
│  Cross-Encoder   │──▶ reranks candidates, trims to final top-k
│    Reranker      │
└─────────────────┘
  │
  ▼
┌─────────────────┐   Jinja2 templates, versioned (rag_v1.md, rag_v2.md, ...)
│  Prompt Manager  │──▶ builds the grounded prompt
└─────────────────┘
  │
  ▼
┌─────────────────┐   OpenAI / Anthropic / Google — pluggable via Protocol
│   LLM Services   │──▶ generates the final answer
└─────────────────┘
  │
  ▼
Answer + logged QueryPerformanceTracker record
```

Every stage logs structured events (latency, counts, cost where applicable) through a shared `observability` layer, and the `RAGChat` orchestrator (`features/rag_chat.py`) ties the whole flow together end-to-end.

## Key Features

- **Sentence-aware chunking** — token-budgeted (`tiktoken`, `cl100k_base`), configurable size/overlap, never splits a sentence, with an oversized-sentence fallback.
- **Local embeddings** — `sentence-transformers` (`all-MiniLM-L6-v2`), batched and normalized, run off the event loop via `asyncio.to_thread`.
- **Pluggable query techniques**
  - *Query Expansion* — LLM generates multiple paraphrased queries to widen recall.
  - *HyDE* — LLM generates a hypothetical answer document, which is embedded/searched instead of (or alongside) the raw query.
- **Pluggable retrieval strategies**
  - *Vector search* — pgvector cosine similarity.
  - *Keyword search* — lexical/full-text matching.
  - *Hybrid search* — runs both concurrently, min-max normalizes each score range, then merges with configurable weights (default `0.6` vector / `0.4` keyword).
- **Cross-encoder reranking** — `cross-encoder/ms-marco-MiniLM-L6-v2` re-scores the merged candidate set for a more precise final top-k.
- **Multi-provider LLM layer** — a common `LLMProvider` protocol implemented for **OpenAI**, **Anthropic**, and **Google**, with structured-output support (Pydantic `response_schema`), streaming, a fan-out `complete_all` mode, and typed exception handling.
- **Versioned prompts** — Jinja2 templates stored as markdown files (`rag_v1.md`, `rag_v2.md`); the active version is pinned in settings so answer quality can be tracked per prompt revision.
- **Multi-tenancy** — every document, chunk, and query is scoped by `tenant_id`.
- **Observability** — structured JSON logging, request-ID context propagation, per-stage latency tracking, embedding cost estimation, and named events.
- **Evaluation loop** — query traces are appended to a JSONL performance log and can be compiled into a versioned evaluation report (`evalution_results/evaluation_<version>.json`) via the `/evalution/report` endpoint.
- **Test coverage** — a unit test mirrors nearly every manager, service, provider, and route; there's also an integration test for keyword retrieval.

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI + Uvicorn |
| Vector store | PostgreSQL + `pgvector` (via `psycopg`/`asyncpg`) |
| Embeddings | `sentence-transformers` (local, `all-MiniLM-L6-v2`) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Tokenization | `tiktoken` |
| LLM providers | OpenAI, Anthropic, Google (Gemini) |
| Prompting | Jinja2 templates |
| Config | `pydantic-settings` |
| Testing | `pytest`, `pytest-asyncio`, `respx` |
| Tooling | `ruff`, `mypy`, `pre-commit`, `uv` |

## Project Structure

```
module-09-rag/
├── main.py                        # placeholder entrypoint
├── docker-compose.yml             # pgvector-enabled Postgres container
├── pyproject.toml                 # dependencies (managed with uv)
├── chunks.txt                     # sample chunked output
├── evalution_results/             # versioned evaluation reports + eval dataset
├── excercise/coding/              # standalone practice exercises
├── tests/                         # unit + integration test suite
└── src/rag_app/
    ├── api/                       # FastAPI app + routers (rag, index, retrive, upsert, evalution)
    ├── core/                      # Settings (pydantic-settings) and env config
    ├── chunking/                  # ChunkingManager
    ├── embedding/                 # EmbeddingManager
    ├── query/                     # QueryManager + techniques (expansion, HyDE)
    ├── retrieval/
    │   ├── candidate/             # vector / keyword / hybrid search strategies
    │   └── re_ranker/             # cross-encoder reranker
    ├── prompts/                   # PromptManager + versioned prompt templates
    ├── providers/                 # OpenAI / Anthropic / Google provider adapters
    ├── services/                  # thin service layer wrapping managers for the API
    ├── features/rag_chat.py       # end-to-end RAG orchestration
    ├── models/                    # Pydantic request/response/domain models
    ├── db/                        # schema.sql + index/upsert/retrieve DB access
    ├── tracker/                   # per-stage performance trackers
    ├── evalution/report.py        # evaluation report generation
    └── observability/             # logging, tracing, metrics, cost, request context
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Service name, version, status |
| `GET` | `/health` | Health check |
| `GET` | `/genrate_uuid` | Generates a UUID (useful for `tenant_id`) |
| `POST` | `/index/process` | Chunk, embed, and index documents for a tenant |
| `POST` | `/upsert/chunks` | Upsert pre-built chunks into the store |
| `POST` | `/retrive/` | Run retrieval only (candidate search + optional rerank) |
| `POST` | `/rag/chat_answer` | Full RAG pipeline: query → retrieve → prompt → LLM answer |
| `GET` | `/evalution/report` | Generate/refresh the evaluation report for the current app version |

## Configuration

All runtime behavior is controlled through `Settings` (`src/rag_app/core/settings.py`), loaded from environment variables / a `.env` file. Notable defaults:

| Setting | Default | Purpose |
|---|---|---|
| `chunking_strategy` / `chunk_size` / `chunk_overlap` | `sentence` / `500` / `50` | Chunking behavior |
| `embedding_model` | `all-MiniLM-L6-v2` | Embedding model |
| `default_query_strategy` | `query_expansion` | `query_expansion`, `query_HyDE`, or `None` |
| `default_retrieval_strategy` | `hybrid_search` | `vector_search`, `keyword_search`, or `hybrid_search` |
| `default_vector_search_weight` / `default_keyword_search_weight` | `0.6` / `0.4` | Hybrid merge weights |
| `re_ranker_availability` | `True` | Toggle cross-encoder reranking |
| `default_llm_provider` / `default_llm_model` | `google` / `gemini-3.5-flash-lite` | Primary LLM |
| `rag_prompt_running_version` | `rag_v2.md` | Active prompt template |

Copy `.env.example` to `.env` and fill in API keys (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`), the Postgres connection details, and tenant API keys before running the service.

## Getting Started

```bash
# 1. Start Postgres with pgvector
docker compose up -d

# 2. Install dependencies (uv is used as the package manager)
uv sync

# 3. Configure environment
cp .env.example .env
# edit .env with your API keys and DB credentials

# 4. Run the API
uv run uvicorn rag_app.api.app:app --reload

# 5. Run tests
uv run pytest
```

Once running, the interactive API docs are available at `http://localhost:8000/docs`.

## Evaluation

Every call to `/rag/chat_answer` writes a `QueryPerformanceTracker` record (query, technique count, chunk IDs used, and the final answer) to a JSONL log. The `/evalution/report` endpoint reads all records matching the current `app_version`, computes aggregate metrics, and writes a versioned report to `evalution_results/evaluation_<version>.json` — allowing retrieval/prompt/model changes to be compared version-over-version against the same evaluation dataset (`evalution_results/evalution_dataset.json`).

## Notes

- The `excercise/coding/` folder contains small, standalone practice snippets (context formatting, HyDE vs. direct-retrieval comparison) separate from the main application — useful for study but not wired into the API.
- The retriever manager's docstring notes that this module's job is to keep the retrieval **interface** stable while the underlying storage implementation evolves (from local/brute-force in earlier modules to pgvector here), which is a useful mental model for how the whole codebase is layered: managers define behavior, services expose it to the API, and providers/strategies are freely interchangeable underneath.