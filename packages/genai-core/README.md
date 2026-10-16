# genai-core

Unified LLM provider and service abstraction extracted and adapted from `module-09-rag` and `module-15-function_calling`.

## Features

- **Unified LLMProvider Protocol**:
  - `complete_structure(prompt, response_schema)`: Structured outputs with schema validation
  - `complete_tool(messages, tools, tool_choice)`: Multi-turn tool calling and function declaration handling
  - `stream(messages)`: Asynchronous streaming of token deltas
- **Normalized Response Model**: `LLMResponseModel` unifying text, parsed structured data (`data`), tool calls (`tool_calls`), model version, latency, and token usage.
- **Provider Adapters**:
  - `GoogleProvider`: Native Gemini support for structured JSON schemas and function calling
  - `OpenAIProvider`: Responses API integration with structured JSON schemas and function tools
  - `AnthropicProvider`: Claude Messages API integration with tool choice and JSON response parsing
- **Unified `LLMService`**: Provider-independent service supporting dependency injection, provider fallback, error wrapping, concurrent calls (`complete_all`), and streaming.

## Installation

```bash
uv pip install -e packages/genai-core
```

## Running Tests

```bash
uv run --directory packages/genai-core pytest
```
