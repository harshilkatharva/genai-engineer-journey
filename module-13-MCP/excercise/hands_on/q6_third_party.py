from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server = StdioServerParameters(
    command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "./sandbox"]
)


async def main():
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            resources = await session.list_resources()

            print("Tools:", tools)
            print("Resources:", resources)
