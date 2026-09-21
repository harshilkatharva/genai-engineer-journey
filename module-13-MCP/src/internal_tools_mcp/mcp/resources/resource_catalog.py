"""Self-contained read-only resource catalog for the MCP server."""

import json

from mcp.server.mcpserver import MCPServer


def _encode(record: dict[str, object]) -> str:
    return json.dumps(record, indent=2, sort_keys=True)


def register_resources(mcp: MCPServer) -> None:
    """Register five independent resources backed by local in-memory data."""

    @mcp.resource(
        "internal://company/profile",
        name="company_profile",
        description="Basic profile for the fictional internal tools company.",
        mime_type="application/json",
    )
    def company_profile() -> str:
        return _encode(
            {
                "company_id": "company-001",
                "name": "Northstar Labs",
                "mission": "Build dependable internal AI tools.",
                "support_email": "support@northstar.example",
            }
        )

    @mcp.resource(
        "internal://catalog/services",
        name="service_catalog",
        description="Catalog of services available to internal teams.",
        mime_type="application/json",
    )
    def service_catalog() -> str:
        return _encode(
            {
                "services": [
                    {
                        "service_id": "search",
                        "name": "Knowledge Search",
                        "owner": "platform",
                    },
                    {
                        "service_id": "answers",
                        "name": "Knowledge Answers",
                        "owner": "applied-ai",
                    },
                    {
                        "service_id": "cases",
                        "name": "Support Cases",
                        "owner": "customer-ops",
                    },
                ]
            }
        )

    @mcp.resource(
        "internal://policies/access",
        name="access_policy",
        description="Read-only access policy for internal tools.",
        mime_type="application/json",
    )
    def access_policy() -> str:
        return _encode(
            {
                "policy_id": "policy-access-001",
                "classification": "internal",
                "rules": [
                    "Use only approved read-only tools.",
                    "Provide a tenant identifier for tenant-scoped searches.",
                    "Do not submit credentials or arbitrary SQL.",
                ],
            }
        )

    @mcp.resource(
        "internal://records/handbook-001",
        name="handbook_record",
        description="One specific handbook record owned by this MCP server.",
        mime_type="application/json",
    )
    def handbook_record() -> str:
        return _encode(
            {
                "record_id": "handbook-001",
                "title": "MCP Integration Guide",
                "status": "published",
                "owner": "platform",
                "summary": "Guidance for exposing safe internal capabilities through MCP.",
            }
        )

    @mcp.resource(
        "internal://status/health",
        name="health_status",
        description="Health status for this standalone MCP resource catalog.",
        mime_type="application/json",
    )
    def health_status() -> str:
        return _encode(
            {
                "service": "internal-tools-mcp",
                "status": "healthy",
                "resource_store": "in-memory",
                "read_only": True,
            }
        )
