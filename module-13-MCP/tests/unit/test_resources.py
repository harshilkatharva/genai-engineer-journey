import json

import pytest
from mcp.server.mcpserver import MCPServer

from internal_tools_mcp.mcp.resources.resource_catalog import register_resources


RESOURCE_URIS = (
    "internal://company/profile",
    "internal://catalog/services",
    "internal://policies/access",
    "internal://records/handbook-001",
    "internal://status/health",
)


@pytest.mark.asyncio
async def test_all_resources_are_registered_and_readable():
    server = MCPServer("test-resources")
    register_resources(server)

    resources = await server.list_resources()
    assert [str(resource.uri) for resource in resources] == list(RESOURCE_URIS)

    for uri in RESOURCE_URIS:
        result = await server.read_resource(uri)
        assert result
        assert json.loads(result[0].content)


@pytest.mark.asyncio
async def test_resource_payloads_have_expected_identity_fields():
    server = MCPServer("test-resources")
    register_resources(server)

    company = await server.read_resource("internal://company/profile")
    health = await server.read_resource("internal://status/health")

    assert json.loads(company[0].content)["company_id"] == "company-001"
    assert json.loads(health[0].content)["read_only"] is True
