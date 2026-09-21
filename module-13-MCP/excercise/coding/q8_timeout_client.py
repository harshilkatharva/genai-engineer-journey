import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="python3",
    args=["excercise/coding/q8_timeout_server.py"],
)


async def call_with_timeout(session, tool_name, arguments, timeout=5):
    try:
        return await asyncio.wait_for(
            session.call_tool(tool_name, arguments),
            timeout=timeout,
        )

    except asyncio.TimeoutError:
        print(f"Tool '{tool_name}' timed out after {timeout} seconds.")
        return None


async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("calling timeout")
            result = await call_with_timeout(
                session,
                "slow_tool",
                {"query": "test"},
                timeout=5,
            )

            if result is not None:
                print("Result:", result)
            else:
                print("Timeout for calling function.")


if __name__ == "__main__":
    asyncio.run(main())
