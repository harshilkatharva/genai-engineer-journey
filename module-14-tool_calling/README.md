# Module 14: Tool Calling

A customer-support agent that exposes a FastAPI endpoint and uses an MCP server for
order, product, and escalation tools. The agent supports Google Gemini, OpenAI, and
Anthropic models through a shared provider interface.

## How It Works

`POST /chat_tool` reads the cancellation policy from the MCP resource
`support://cancellation-policy`, discovers available MCP tools, and runs a bounded
model/tool loop. The API starts the MCP server as a child process using the MCP
stdio transport; PostgreSQL backs order, product, and escalation operations. The
policy resource is read from `data/daily_deal_product_cancellation_policy.txt`.

## Requirements

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/)
- PostgreSQL for order, product, and escalation operations
- An API key for the selected LLM provider

## Setup

From the repository root, install the project and development dependencies:

```bash
uv sync --dev
```

Create a local environment file:

```bash
cp .env.example .env
```

Edit `.env` and set the API key for the selected provider. `DEFAULT_LLM_PROVIDER`
accepts `google`, `openai`, or `anthropic`; `DEFAULT_LLM_MODEL` selects the model
for that provider. Set `DATABASE_URL` to your PostgreSQL connection string. The
provided `.env.example` contains the full list of supported settings and example
values.

The database must contain the `orders`, `products`, and `support_escalations`
tables. The checked-in schema creates these tables:

```bash
psql "$DATABASE_URL" -f src/customer_support_agent/db/schema.sql
```

The schema does not add sample orders or products; populate the tables with the data
you want the tools to retrieve.

## Start the API and MCP Server

Run this single command from the repository root:

```bash
uv run uvicorn customer_support_agent.api:app --reload
```

This starts the API at `http://127.0.0.1:8000`. You do not need a second terminal
for MCP: when `/chat_tool` is called, the API launches
`customer_support_agent.mcp_server.server` as a stdio subprocess and closes it when
the request finishes. The MCP server is not a separate HTTP service.

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Chat API

Send at least one message to `POST /chat_tool`. `customer_id` is optional and is
passed to the escalation tool when a support ticket is created.

```bash
curl -X POST http://127.0.0.1:8000/chat_tool \
	-H 'Content-Type: application/json' \
	-d '{
		"messages": [{"role": "user", "content": "Where is order ORD-123?"}],
		"customer_id": "customer-1"
	}'
```

The response contains the assistant text, number of model iterations, whether the
iteration limit was exhausted, and an optional error code:

```json
{
	"text": "...",
	"iterations": 2,
	"exhausted": false,
	"error": null
}
```

## MCP Tools

| Tool | Purpose |
| --- | --- |
| `lookup_order` | Look up an order by order ID. |
| `product_search` | Search products by description and optional maximum price. |
| `product_details` | Retrieve a product by product ID. |
| `escalation` | Create a human-support ticket with a priority of `low`, `normal`, `high`, or `urgent`. |

The `support://cancellation-policy` resource supplies the current cancellation
policy as context to the model; there is no cancellation-policy tool call.

## Configuration and Logging

Settings are loaded from environment variables or `.env`. Important settings are:

| Setting | Purpose |
| --- | --- |
| `DEFAULT_LLM_PROVIDER` | Provider to use: `google`, `openai`, or `anthropic`. |
| `DEFAULT_LLM_MODEL` | Model name passed to the selected provider. |
| `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | Provider credentials; set the key for the selected provider. |
| `DATABASE_URL` | PostgreSQL connection string. |
| `MAX_TOOL_ITERATIONS` | Maximum model/tool rounds; defaults to `5`. |
| `LOG_LEVEL` | Application log level. |
| `TOOL_CALL_LOG_FILE` | JSON Lines log destination; defaults to `logs/tool_calls.jsonl`. |

Structured JSON events are written to standard output and the configured log file.
Tool events include the tool name, arguments, duration, outcome, and request ID.

## Development

Run lint and the unit test suite:

```bash
uv run ruff check .
uv run pytest
```

Integration tests use a real configured LLM provider and PostgreSQL-backed tools.
They are skipped unless explicitly enabled:

```bash
INTEGRATION_TEST=1 uv run pytest -m integration
```
