# Internal Tools MCP

An MCP server that exposes read-only internal capabilities through Module 8 vector retrieval and Module 9 RAG chat. The server supports local stdio execution and remote-style SSE or streamable HTTP transports.

## Capabilities

### Tools

| Tool | Purpose | Backend |
| --- | --- | --- |
| `module9_chat_answer` | Answer a question for a tenant | Module 9 `RAGChat` |
| `module8_retrieve_documents` | Retrieve matching documents | Module 8 `RetriveServiceManager` |

All tool inputs are validated before the underlying service is called. Tool invocations are logged with the tool name, arguments, and caller. The tools are read-only and do not accept arbitrary SQL or write operations.

`module9_chat_answer` accepts:

```json
{
	"query": "What was the name of the ship arriving at Marseilles on February 24, 1815?",
	"tenant_id": "06f197fb-3b03-469f-b3ba-461dae52cf7a"
}
```

`module8_retrieve_documents` accepts:

```json
{
	"tenant_id": "06f197fb-3b03-469f-b3ba-461dae52cf7a",
	"query": "vector retrieval",
	"document_type": null,
	"top_k": 5
}
```

### Resources

The resources are independent, read-only in-memory records owned by this MCP project. They do not import Module 8 or Module 9 later we can add with our database.

- `internal://company/profile`
- `internal://catalog/services`
- `internal://policies/access`
- `internal://records/handbook-001`
- `internal://status/health`

## Project Layout

```text
src/internal_tools_mcp/
	server.py                         MCP composition root and transport startup
	integrations.py                   Lazy sibling-module imports
	rag_service/rag_service.py        Module 9 adapter
	retrive_service/retrive_service.py Module 8 adapter
	mcp/tools/tools.py                Tool registration and validation
	mcp/resources/resource_catalog.py Standalone resource registration
client.py                            Stdio capability-discovery client
tests/unit/                           Unit tests with injected service doubles
```

## Requirements

- Python 3.12 or newer
- `uv`
- The sibling projects in the handbook repository:
	- `../module-08-VectorDB`
	- `../module-09-rag`
- Runtime configuration required by those projects, including their database and LLM provider settings

The MCP server itself uses the MCP 2.x API:

```python
from mcp.server.mcpserver import MCPServer
```

## Setup

From this project directory:

```bash
uv sync
```

The adapters resolve the sibling source directories from the repository layout. Keep this project beside `module-08-VectorDB` and `module-09-rag` as shown:

```text
genai-engineer-journey/
	module-08-VectorDB/
	module-09-rag/
	module-13-MCP/
```

## Run Locally With Stdio

Start the server manually:

```bash
uv run python -m internal_tools_mcp.server --transport stdio
```

For a complete local demonstration, run the client in another terminal:

```bash
uv run python client.py
```

The client initializes a session, discovers tools and resources, reads every resource, and invokes both tools.

Do not write ordinary diagnostic output to server stdout when using stdio. stdout is reserved for MCP JSON-RPC messages; diagnostics must go to stderr or through logging.

## Run With SSE

HTTP and SSE transports require a shared bearer token. Set it with an environment variable or CLI flag:

```bash
export MCP_API_TOKEN="replace-with-long-random-secret"

uv run python -m internal_tools_mcp.server \
	--transport sse \
	--host 127.0.0.1 \
	--port 8000
```

Clients send:

```http
Authorization: Bearer replace-with-long-random-secret
```

The SSE endpoint is:

```text
http://127.0.0.1:8000/sse
```

## Run With Streamable HTTP

Start the server with the same token requirement:

```bash
export MCP_API_TOKEN="replace-with-long-random-secret"

uv run python -m internal_tools_mcp.server \
	--transport streamable-http \
	--host 127.0.0.1 \
	--port 8000
```

The MCP endpoint is:

```text
http://127.0.0.1:8000/mcp
```

Unauthenticated requests receive a `401` response with a `WWW-Authenticate: Bearer` header, and the server rejects invalid tokens before any MCP request is processed.

## Additional Security Before External Exposure

This server is intentionally read-only and token-protected, but before exposing it beyond a trusted internal network, add the following controls from the module basics:

- TLS termination at a reverse proxy or ingress, so traffic is encrypted in transit and the server is not directly internet-facing.
- Mutual TLS (mTLS) or a trusted identity gateway if the deployment is multi-service or crosses network boundaries.
- A proper OAuth or signed JWT flow instead of a shared static secret when clients are not all fully trusted internal processes.
- IP allowlisting, VPC/private-network boundaries, and firewall rules to restrict access to known internal subnets or a bastion path.
- Rate limiting, request-size caps, and request logging/monitoring to detect abuse, token leakage, and anomalous traffic.
- DNS rebinding protection and host allowlisting for the transport security options, especially when the service is behind a browser-facing proxy.
- Rotation and secret management for the API token via environment injection or a secret manager, rather than hard-coded values in source control.
- Least-privilege network design: keep the server on a private segment, restrict outbound access to only required dependencies, and avoid exposing admin or debugging routes.

These layers are the minimum needed before moving from a trusted internal MCP endpoint to a broader network or internet exposure.

## Tests

Run the maintained MCP test suite:

```bash
uv run pytest -q
```

The tests use injected fake services for Module 8 and Module 9, so unit tests do not require PostgreSQL, an LLM API key, downloaded embedding models, or network access.

## Architecture

```mermaid
flowchart LR
		Client[MCP client] --> Transport[stdio / SSE / streamable HTTP]
		Transport --> Server[MCP server]
		Server --> Tools[Registered tools]
		Server --> Resources[Standalone resources]
		Tools --> RAG[Module 9 RAGChat adapter]
		Tools --> Vector[Module 8 retrieval adapter]
		RAG --> RAGApp[Module 9 application]
		Vector --> VectorApp[Module 8 application]
```

## Notes

- The adapters preserve the existing Module 8 and Module 9 service APIs rather than copying their implementations.
- The Module 9 adapter creates an observability request context and redirects legacy stdout diagnostics to stderr for stdio protocol safety.
- The Module 8 adapter also redirects legacy stdout diagnostics to stderr.
- Resource data is intentionally local and deterministic for capability discovery and testing.
