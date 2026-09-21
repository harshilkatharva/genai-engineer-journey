from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(command="python", args=["mcp_server.py"])


async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print(await session.list_tools())
            print(await session.list_resources())

            result = await session.call_tool("add_numbers", {"a": 10, "b": 20})
            print(result)

            content = await session.read_resource("greeting://hello")
            print(content)
