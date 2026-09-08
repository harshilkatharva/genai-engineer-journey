# LangChain Model Test Comparison

## Chain Tested

Both suites test the migrated `RAGChatLangchain` chain:

`retrieval -> context formatting -> prompt -> chat model -> PydanticOutputParser`

The retriever and Redis history factory are isolated in both suites so the comparison focuses on the chat model and chain output behavior.

## Fast Fake-Model Suite

File: `tests/unit/features/test_rag_chat_langchain_fake_model.py`

The suite uses LangChain Core's `FakeListChatModel`. It is fast, deterministic, and makes no provider or Redis network calls. It verifies that:

- JSON model output is parsed into `RAGResposne`.
- The response contains the expected `text` and `source` fields.
- Repeated calls consume predictable fake responses.
- The chain can run with the configured message-history interface mocked.

Run it with:

```bash
uv run pytest -q tests/unit/features/test_rag_chat_langchain_fake_model.py
```

## Real-Provider Integration Suite

File: `tests/integration/llm/test_rag_chat_langchain_real.py`

The suite uses the configured `ChatGoogleGenerativeAI` model and the real provider API. Retrieval and Redis history are still isolated so the test does not depend on the local database or a live Redis conversation.

It is skipped by default. Run it explicitly with:

```bash
RUN_REAL_LLM_TESTS=1 uv run pytest -q tests/integration/llm/test_rag_chat_langchain_real.py
```

A valid `GOOGLE_API_KEY` must also be available through the environment or `.env`.

## Fake Versus Real Behavior

- The fake model returns the exact strings supplied in `responses`. This makes exact response assertions appropriate for unit tests.
- The real provider returns provider-generated chat messages and is not deterministic. The integration test therefore checks the stable contract, such as `RAGResposne` shape and non-empty answer text, rather than one exact sentence.
- In live testing, the provider returned structured JSON wrapped in a Markdown code fence, while the fake model returned plain JSON. The existing `PydanticOutputParser` accepted the provider response, so the fake test should not assume that real output has no formatting wrapper.
- The real model return `source: null or None` when the prompt does not require or support a source. The fake suite can assert a specific source value, but the integration suite allows `None`, a string, or a list.
- Fake responses are consumed in a fixed sequence. Real responses depend on provider generation, prompt wording, model version, and service configuration.

## Result

The fake suite is the default regression safety net. The real-provider suite is a smoke test for provider compatibility and parser behavior, and should be run deliberately because it requires credentials, network access, time, and API cost.
