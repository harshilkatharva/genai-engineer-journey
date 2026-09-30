# Multi-Provider Function Calling Library

A Python library for sending chat requests to Google Gemini, OpenAI, or Anthropic through one service API. It normalizes text and function-call responses, builds provider tool schemas from Pydantic models, validates tool arguments, and can extract validated structured output through a forced function call.

## Requirements

- Python 3.12 or newer
- An API key for the provider you plan to call
- `uv` for the repository's development workflow, or `pip` to install the package

## Install

From a checkout of this repository:

```bash
uv sync --all-groups
```

Alternatively, install the package and its runtime dependencies with pip:

```bash
python -m pip install .
```

## Provider Configuration

The library reads settings from environment variables and a `.env` file in the current working directory. Create `.env` with the key for the provider you want to use:

```dotenv
DEFAULT_LLM_PROVIDER=google
DEFAULT_LLM_MODEL=gemini-3.5-flash-lite
DEFAULT_LLM_TEMPERATURE=0.2

GOOGLE_API_KEY=your-google-api-key
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

Provider names are `google`, `openai`, and `anthropic`. You can set more than one API key and choose the provider per request. If `provider` is omitted, `DEFAULT_LLM_PROVIDER` is used.

`DEFAULT_LLM_MODEL` is shared by the providers. Choose a model supported by the selected provider. When the configured model name does not match that provider's expected prefix, the adapter falls back to its built-in default: `gemini-3.5-flash-lite`, `gpt-4o-mini`, or `claude-3-5-sonnet-20241022`.

The provider SDK clients are created by `LLMService`; the service selects the configured provider when a request is made. Keep API keys in environment configuration and do not commit `.env`.

## Basic Completion

Create an `LLMService`, then pass an `LLMManagerRequest` with a list of `ChatMessage` objects:

```python
import asyncio

from function_calling_library.models import ChatMessage, LLMManagerRequest
from function_calling_library.services import LLMService


async def main() -> None:
	service = LLMService()
	response = await service.complete(
		LLMManagerRequest(
			provider="google",  # Optional; defaults to DEFAULT_LLM_PROVIDER.
			messages=[ChatMessage(role="user", content="Explain function calling briefly.")],
		)
	)

	if response.error:
		print(f"{response.error.provider}: {response.error.message}")
		return

	print(response.text)
	print(response.model, response.usage)


asyncio.run(main())
```

`LLMManagerResponse` contains `text`, `tool_calls`, `model`, token `usage`, optional `raw_response`, and optional `error`. Provider failures and unsupported provider names are represented by `error`; successful completions leave it as `None`. The raw response is provider-SDK-specific.

## Models and Tool Calls

The main request/response models are:

- `ChatMessage`: message role and text, with optional `tool_call_id` and `tool_name` fields.
- `LLMManagerRequest`: provider, messages, tool specifications, and tool-choice mode.
- `LLMManagerResponse`: generated text and/or normalized `ToolCall` items.
- `ToolCall`: provider call ID, registered tool name, and raw arguments.
- `LLMError`: provider, error code/message, optional HTTP status, and retryability.

The service does not execute a requested tool automatically. The model returns a `ToolCall`; your application decides whether it is allowed and then executes the matching local handler.

### Define and execute a tool

Define one Pydantic arguments model and one `ToolSpec` for each tool. The `ToolSpec` schema adapters convert that model's JSON Schema to each provider's tool declaration. Its `handler` is local Python code: it is never sent to the provider.

```python
import asyncio

from pydantic import BaseModel

from function_calling_library.models import ChatMessage, LLMManagerRequest, ToolSpec
from function_calling_library.services import LLMService, execute_tool_calls


class WeatherArguments(BaseModel):
	city: str


async def get_weather(arguments: WeatherArguments) -> dict[str, str]:
	# Replace this example with your own application or API call.
	return {"city": arguments.city, "forecast": "sunny"}


weather_tool = ToolSpec(
	name="get_weather",
	description="Get the current weather for a city.",
	argument_model=WeatherArguments,
	handler=get_weather,
)


async def main() -> None:
	service = LLMService()
	response = await service.complete(
		LLMManagerRequest(
			provider="google",
			messages=[ChatMessage(role="user", content="What's the weather in Paris?")],
			tools=[weather_tool],
		)
	)

	if response.error:
		print(response.error.message)
		return

	if not response.tool_calls:
		print(response.text)
		return

	results = await execute_tool_calls(response.tool_calls, [weather_tool])
	for result in results:
		if result.error:
			print(f"{result.call_id}: {result.error.message}")
		else:
			print(f"{result.call_id}: {result.result}")


asyncio.run(main())
```

`execute_tool_calls(calls, tools)` matches calls to specs by tool name, validates arguments using the Pydantic model, and invokes synchronous or asynchronous handlers concurrently. Returned `ToolExecutionResult` objects preserve each original `call_id` and input order. Unknown tools, invalid arguments, missing handlers, and handler failures are returned as structured `error` values instead of being raised by the executor.

The current library stops after local tool execution. It does not yet provide a provider-independent follow-up turn that submits tool results to the model and obtains a final answer. In particular, tool-result message formatting is provider-specific; applications should not assume that adding a `ChatMessage(role="tool", ...)` works identically across all adapters.

### Tool-choice modes

Set `tool_choice` on `LLMManagerRequest` to control provider behavior:

```python
from function_calling_library.models import ToolChoice

automatic = ToolChoice(mode="auto")
require_a_tool = ToolChoice(mode="any")
force_weather = ToolChoice(mode="specific", tool_name="get_weather")
```

`auto` lets the provider choose between text and a function call, `any` requires a function call, and `specific` forces the named function. Provider APIs express these modes differently; the adapters translate the shared mode to the selected SDK.

## Structured Output

Use `extract_structured_output` when the desired result is a Pydantic object. The helper creates a temporary tool from the output model, forces that function call, and validates the returned arguments:

```python
import asyncio

from pydantic import BaseModel

from function_calling_library.models import ChatMessage
from function_calling_library.services import LLMService, extract_structured_output


class Answer(BaseModel):
	answer: str
	confidence: float


async def main() -> None:
	answer, error = await extract_structured_output(
		LLMService(),
		[ChatMessage(role="user", content="Answer the question and estimate confidence.")],
		Answer,
		provider="google",
	)
	if error:
		print(error.message)
	else:
		print(answer)


asyncio.run(main())
```

The return value is `(parsed_model, None)` on success or `(None, ToolArgumentError)` when the provider fails, does not return the forced call, or returns arguments that fail model validation.

## Streaming

`LLMService.stream(provider_name, messages)` yields text chunks:

```python
from function_calling_library.models import ChatMessage
from function_calling_library.services import LLMService


async def stream_answer() -> None:
	async for chunk in LLMService().stream(
		"google", [ChatMessage(role="user", content="Tell me a short story.")]
	):
		print(chunk, end="", flush=True)
```

Streaming is text-only in the current adapters and does not use the structured `LLMError` return shape used by `complete`.

## Tests

Run the default unit suite; integration-marked tests are skipped unless explicitly enabled:

```bash
pytest
```

Run integration tests that call configured external providers:

```bash
export INTEGRATION_TEST=1
pytest tests/integration -m integration
```

Integration tests require usable API keys and may incur provider charges. For example, live provider tool-choice checks are included for Google and OpenAI. Clear the variable to return to the default behavior:

```bash
unset INTEGRATION_TEST
```
