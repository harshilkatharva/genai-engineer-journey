import asyncio
import json
from datetime import datetime, timezone

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


LOG_FILE = "excercise/coding/logs/tool_calls.jsonl"

server_params = StdioServerParameters(
    command="python3",
    args=["excercise/coding/q5_least_privilege_server.py"],
)


def write_log(tool_name, arguments, result, status):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "arguments": arguments,
        "result": result,
        "status": status,
    }

    with open(LOG_FILE, "a") as file:
        file.write(json.dumps(entry, default=str) + "\n")


async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            arguments = {"employee_id": "EMP-001"}

            try:
                result = await session.call_tool(
                    "get_employee",
                    arguments,
                )

                status = "error" if result.is_error else "success"

                write_log(
                    "get_employee",
                    arguments,
                    result.content,
                    status,
                )

            except Exception as exc:
                write_log(
                    "get_employee",
                    arguments,
                    str(exc),
                    "error",
                )

                raise


if __name__ == "__main__":
    asyncio.run(main())
