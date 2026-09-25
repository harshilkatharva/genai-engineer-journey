# Module 14: Tool calling

Phase 2 adds a customer-support tool loop around the Phase 1 provider and database
boundaries. The application exposes `POST /chat_tool` and uses an MCP client over a
stdio subprocess. The MCP server owns database-backed tool execution, while the
cancellation policy is exposed as the `support://cancellation-policy` MCP resource.

## Run

```bash
uv run uvicorn customer_support_agent.api:app --reload
```

The request contains OpenAI-style messages:

```json
{"messages": [{"role": "user", "content": "Where is order 123?"}], "customer_id": "c-1"}
```

The MCP server exposes `lookup_order`, `product_search`, `product_details`, and
`escalation`. `run_tool_chat` repeatedly asks the provider for tool calls through MCP,
executes them, and appends tool results to the conversation. It enforces
`MAX_TOOL_ITERATIONS` (default five), converts tool failures into safe tool results,
and returns a friendly message for provider failures or an exhausted loop.
Every tool description tells the model what the tool does, when to use it, and when
not to use it.

## Development

```bash
uv run ruff check .
uv run pytest
INTEGRATION_TEST=1 uv run pytest -m integration
```

Integration tests are opt-in because they call the configured real LLM provider and
real PostgreSQL-backed tools. They run only when `INTEGRATION_TEST=1` is exported:

```bash
export INTEGRATION_TEST=1
uv run pytest -m integration
```

Configure `DATABASE_URL`, the selected provider API key, and the provider/model in
`.env` before running them. Unit tests remain isolated from external services.
