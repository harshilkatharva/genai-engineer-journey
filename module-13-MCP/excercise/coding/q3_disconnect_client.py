import asyncio
import logging

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO)

server_params = StdioServerParameters(
    command="python3",
    args=["excercise/coding/paginated_resource_server.py"],
)


async def main():
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                print("Connected to MCP server.")

                # First request
                result = await session.read_resource("records://page/1")
                print(result)

                # the exception will be caught below.
                result = await session.read_resource("records://page/0")
                print(result)

    except (ConnectionError, BrokenPipeError, EOFError) as exc:
        logging.error("MCP server disconnected: %s", exc)

    except Exception as exc:
        logging.error("MCP session failed: %s", exc)

    finally:
        print("MCP client shutting down gracefully.")


if __name__ == "__main__":
    asyncio.run(main())
