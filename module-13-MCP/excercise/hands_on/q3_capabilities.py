from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(command="python", args=["mcp_server.py"])


async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            resources = await session.list_resources()

            print("=== Tools ===")
            for tool in tools.tools:
                print(tool.name, ":", tool.description)

            print("\n=== Resources ===")
            for resource in resources.resources:
                print(resource.uri, ":", resource.name)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
