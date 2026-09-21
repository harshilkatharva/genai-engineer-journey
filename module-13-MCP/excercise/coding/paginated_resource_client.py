import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


server_params = StdioServerParameters(
    command="python3",
    args=["excercise/coding/paginated_resource_server.py"],
)


async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Read page 1
            result = await session.read_resource("records://page/1")

            print("Page 1:")
            print(result)

            # Read page 2
            result = await session.read_resource("records://page/2")

            print("\nPage 2:")
            print(result)


if __name__ == "__main__":
    asyncio.run(main())
