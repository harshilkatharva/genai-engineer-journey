# LangChain Migration Comparison

## What Became Simpler

- Chain composition is clearer: retrieval, prompt formatting, the chat model, and output parsing are represented as one runnable pipeline.
- `PydanticOutputParser` makes structured LLM responses explicit and validates the final response against the application model.
- `RunnableWithMessageHistory` provides a standard place to attach conversation history to the chain.
- LangChain `Document` gives retrieved text and metadata a common interface for prompt construction.
- The same runnable structure is easy to unit test with stub retrievers and mocked chat models.

## What Became More Complex

- Data flow is less direct. Values move through runnable mappings and configuration dictionaries instead of ordinary function arguments.
- History requires more infrastructure: Redis Stack, authentication, a search index, session-key conventions, and compatibility handling for `langchain-redis`.
- Debugging is less transparent because failures can occur inside runnable configuration, callbacks, parsers, or provider adapters. A simple request may cross several abstractions before reaching retrieval or the model.
- Structured output adds another contract to maintain: the Pydantic model, prompt format instructions, provider behavior, and parser must agree.
- Dependency behavior matters more. This system encountered Redis Search requirements and a `langchain-redis`/RedisVL index-metadata compatibility issue.

## Recommendation For This System

I recommend keeping LangChain for the chat-facing orchestration layer, but not replacing the existing retrieval and database components wholesale.

This system benefits from LangChain where it provides concrete value: composing the RAG pipeline, integrating chat models, parsing structured output, and attaching conversation history. The existing custom retrieval code already contains important application-specific behavior, including tenant filtering, hybrid search, reranking, tracking, and cost/latency observability. Rewriting those parts around LangChain would add abstraction without removing much complexity.

The practical recommendation is a hybrid architecture:

1. Keep custom retrieval, indexing, tracking, and domain models as the source of truth.
2. Use LangChain at the boundary where it improves composition and provider integration.
3. Keep Redis history behind a small application-owned factory so LangChain or Redis library changes do not spread through the application.
4. Add integration tests for Redis and provider behavior in addition to fast mocked unit tests.

For this specific system, the migration is worthwhile, but as a focused orchestration upgrade rather than a full rewrite.
