import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Client-side capability allowlist
ALLOWED_TOOLS = {
    "get_employee",
    "search_documents",
}


async def call_allowed_tool(session, tool_name, arguments):
    # Check capability before invoking the server
    if tool_name not in ALLOWED_TOOLS:
        raise PermissionError(f"Tool '{tool_name}' is not allowed by the client.")

    return await session.call_tool(tool_name, arguments)


server_params = StdioServerParameters(
    command="python3",
    args=["excercise/coding/q10_allowlist_server.py"],
)


async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discover server capabilities
            tools = await session.list_tools()

            print("Server tools:")
            for tool in tools.tools:
                print(f" - {tool.name}")

            # Allowed tool
            result = await call_allowed_tool(
                session,
                "get_employee",
                {"employee_id": "EMP-001"},
            )

            print("Result:", result)

            # Blocked tool
            try:
                await call_allowed_tool(
                    session,
                    "delete_employee",
                    {"employee_id": "EMP-001"},
                )
            except PermissionError as error:
                print("Blocked:", error)


if __name__ == "__main__":
    asyncio.run(main())
