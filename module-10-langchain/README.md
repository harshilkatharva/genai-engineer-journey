# Module 10: LangChain RAG

This module takes the multi-tenant RAG service from [Module 9](../module-09-rag/README.md) and connects its retrieval and LLM components with LangChain. The application exposes the chains through FastAPI; the standalone exercises in `excercise/` demonstrate LangChain prompts, retrievers, output parsers, and message history.

## What Changed From Module 9

Module 9 already contains the RAG capabilities: document ingestion, chunking, embeddings, vector/keyword/hybrid retrieval, reranking, versioned prompts, multiple LLM providers, and evaluation. Module 10 keeps those components and changes how the chat and structured-output flows are assembled.

| Module 9 | Module 10 |
| --- | --- |
| `RAGChat` explicitly calls query processing, retrieval, prompt rendering, and LLM completion in sequence. | `RAGChatLangchain` composes the chat flow as LangChain runnables (LCEL). |
| Prompt templates are rendered into strings and passed to the LLM service. | `ChatPromptTemplate` builds messages, and the configured provider supplies a LangChain chat model. |
| The chat retriever is called directly by the orchestrator. | `LangchainRetriever` adapts the existing retrieval manager to LangChain's `BaseRetriever`. It still uses the configured query strategy, search strategy, and reranker. |
| Chat requests do not add LangChain message-history handling to the RAG flow. | `RunnableWithMessageHistory` loads and saves per-session history in Redis, scoped by tenant and session. |
| The regular chat response is assembled by the feature code. | `PydanticOutputParser` validates the model response as the RAG response schema. |

The main chat chain is assembled in `src/rag_app/features/rag_chat_langchain.py`:

```text
{query, context}
    │
    ├── query ──> LangchainRetriever ──> formatted document context
    │                                      │
    └──────────────────────────────────────┘
                         │
       ┌── RunnableWithMessageHistory ─────┐
       │                                   │
       │  ChatPromptTemplate ──> Chat model│
       │                                   │
       └───────────────────────────────────┘
                         │
               PydanticOutputParser
                         │
                 RAG response model
```

The chain is asynchronous. The application passes the tenant ID as runnable metadata so the retriever only searches that tenant's indexed chunks. Its Redis history key also includes the tenant ID and session ID.

## Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management
- Docker with Docker Compose
- PostgreSQL with the `pgvector` extension and Redis Stack
- API credentials for the LLM provider you intend to use

The default configuration uses Google Gemini (`GOOGLE_API_KEY`). The current configuration module also reads the OpenAI, Anthropic, tenant, and LangSmith environment variables at application import time, so define them even if you do not plan to use those providers. Use your own credentials; do not commit `.env`.

## Configure And Start

Run these commands from the repository root.

1. Install dependencies and create the virtual environment:

   ```bash
   uv sync
   ```

2. Create `.env` with the following variables. Replace each placeholder with an appropriate value. The database URL below matches this repository's Compose configuration.

   ```dotenv
   GOOGLE_API_KEY=replace-with-your-google-key
   OPENAI_API_KEY=replace-with-your-openai-key
   ANTHROPIC_API_KEY=replace-with-your-anthropic-key

   DATABASE_CONNECTION_CONVERSATION_URL=postgresql://postgres:postgres@127.0.0.1:5432/ai_search

   X_API_KEY_Tenant_A=replace-with-tenant-a-key
   X_API_KEY_Tenant_B=replace-with-tenant-b-key

   LANGSMITH_API_KEY=replace-with-your-langsmith-key
   LANGSMITH_PROJECT=module-10-langchain
   LANGSMITH_TRACING=false
   LANGSMITH_ENDPOINT=https://api.smith.langchain.com

   REDIS_URL=redis://default:redis_password@127.0.0.1:6379
   ```

   `REDIS_URL` is optional when using the default Compose password. You can change model/provider and other runtime settings with environment variables; see `src/rag_app/core/settings.py`.

3. Create the external Docker volume expected by `docker-compose.yml` if it does not already exist, then start PostgreSQL and Redis:

   ```bash
   docker volume create module-08-vectordb_postgres_data
   docker compose up -d
   ```

4. Enable pgvector and create the chunk table. The schema is not created automatically by the application:

   ```bash
   docker compose exec -T postgres psql -U postgres -d ai_search \
     -c 'CREATE EXTENSION IF NOT EXISTS vector;'
   docker compose exec -T postgres psql -U postgres -d ai_search \
     < src/rag_app/db/schema.sql
   ```

5. Start the API:

   ```bash
   uv run uvicorn rag_app.api.app:app --reload
   ```

   The API is at `http://localhost:8000`; interactive request schemas are at `http://localhost:8000/docs`. Check readiness at `http://localhost:8000/health`.

## Run The RAG Chat Chain

Index at least one document for a tenant before asking a question. The `documents` field accepts raw text or a path to a text file. The example uses multiline text so it is treated as document content. Keep the same tenant UUID in both requests.

```bash
curl -X POST http://localhost:8000/index/process \
  -H 'Content-Type: application/json' \
  -d '{
    "tenant_id": "550e8400-e29b-41d4-a716-446655440001",
    "documents": ["LangChain composes language model applications from reusable components. Retrieval-augmented generation finds relevant documents and uses them as context for an answer.\nLCEL expresses chains by composing runnable components."],
    "documents_type": ["text"],
    "meta_data": [{}]
  }'
```

Then invoke the chain through the chat endpoint:

```bash
curl -X POST http://localhost:8000/rag/chat_answer \
  -H 'Content-Type: application/json' \
  -d '{
    "tenant_id": "550e8400-e29b-41d4-a716-446655440001",
    "session_id": "demo-session",
    "query": "How does LCEL help build a RAG application?"
  }'
```

The response contains `text` and, when provided by the model, `source`. Reusing `session_id` continues the Redis-backed conversation; use a new session ID to start a separate chat. Embedding and reranking models may need to download on their first run.

### Invoke The Chain From Python

`RAGChatLangchain.get_chat_answer` supplies the required tenant metadata and session configuration for you. Call it from an async context after indexing documents and starting both backing services:

```python
from uuid import UUID

from rag_app.features.rag_chat_langchain import RAGChatLangchain
from rag_app.models import RAGRequest


async def ask() -> None:
	chat = RAGChatLangchain()
	answer = await chat.get_chat_answer(
		RAGRequest(
			tenant_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
			session_id="demo-session",
			query="How does LCEL help build a RAG application?",
		)
	)
	print(answer.text)
```

Run direct Python examples with the project environment, for example from an async script using `uv run python your_script.py`. The LangChain chain itself is available as `chat.chain`; direct calls must pass the tenant ID in runnable metadata and a tenant-scoped session ID in runnable configuration:

```python
tenant_id = UUID("550e8400-e29b-41d4-a716-446655440001")
answer = await chat.chain.ainvoke(
	{"query": "How does LCEL help build a RAG application?"},
	config={
		"metadata": {"tenant_id": tenant_id},
		"configurable": {"session_id": f"{tenant_id}:demo-session"},
	},
)
```

## Other Chains And Exercises

The same API also exposes structured-output chains:

```bash
curl -X POST http://localhost:8000/rag/classification \
  -H 'Content-Type: application/json' \
  -d '{"text": "The product arrived broken and customer support has not replied."}'

curl -X POST http://localhost:8000/rag/extraction \
  -H 'Content-Type: application/json' \
  -d '{"text": "Contact Alex Chen at alex@example.com about order 4312."}'
```

These use `ChatPromptTemplate | chat model | PydanticOutputParser`. To work through the LangChain exercises, open `excercise/hands_on/excercise.ipynb` in VS Code, select the environment created by `uv sync` as the notebook kernel, and run the cells in order. The coding exercises and their datasets/prompts are under `excercise/coding/`.

## Useful Commands

```bash
# Run unit tests (integration tests are excluded by the project configuration)
uv run pytest

# Run Ruff
uv run ruff check .

# Stop the local dependencies
docker compose down
```

## Project Map

```text
src/rag_app/
├── api/                 FastAPI app and endpoint routers
├── features/            LangChain RAG, classification, and extraction chains
├── retrieval/           Search strategies, reranker, and LangChain retriever
├── prompts/             Versioned prompts and ChatPromptTemplate builders
├── providers/           Google, OpenAI, and Anthropic chat model adapters
├── services/            Indexing, retrieval, LLM, and Redis history services
├── db/                  PostgreSQL operations and schema
└── core/                Environment-backed settings

excercise/
├── hands_on/            LangChain notebook exercises
└── coding/              Structured-output exercises and test data
```
