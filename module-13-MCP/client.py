"""Discover and call the Module 8 and Module 9 MCP tools and resources."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    root = Path(__file__).resolve().parent
    source_path = str(root / "src")
    python_path = os.pathsep.join(
        filter(None, [source_path, os.environ.get("PYTHONPATH", "")])
    )
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "internal_tools_mcp.server"],
        cwd=root,
        env={**os.environ, "PYTHONPATH": python_path},
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("Discovered tools:", [tool.name for tool in tools.tools])
            resources = await session.list_resources()
            print(
                "Discovered resources:",
                [str(resource.uri) for resource in resources.resources],
            )

            for resource in resources.resources:
                print(
                    f"Resource {resource.uri}:",
                    await session.read_resource(resource.uri),
                )

            calls = (
                (
                    "Module 9 answer",
                    "module9_chat_answer",
                    {
                        "query": "What was the name of the ship arriving at Marseilles on February 24, 1815?",
                        "tenant_id": "06f197fb-3b03-469f-b3ba-461dae52cf7a",
                    },
                ),
                (
                    "Module 8 documents",
                    "module8_retrieve_documents",
                    {
                        "tenant_id": "06f197fb-3b03-469f-b3ba-461dae52cf7a",
                        "query": "vector retrieval",
                        "top_k": 5,
                    },
                ),
            )
            for label, tool_name, arguments in calls:
                try:
                    result = await session.call_tool(tool_name, arguments)
                    print(f"{label}:", result)
                except Exception as error:
                    print(f"{label} failed: {type(error).__name__}: {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    asyncio.run(main())
