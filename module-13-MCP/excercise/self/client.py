import anyio

from mcp import Client, StdioServerParameters

server = StdioServerParameters(
    command="python3",
    args=["excercise/self/server.py"],
)


async def main():
    async with Client("http://localhost:8000/mcp") as client:
        print("Server:")
        print(client.server_info)

        print("Capabilities:")
        print(client.server_capabilities)

        print("Protocol:")
        print(client.protocol_version)
        print("\n\n")

        tools = await client.list_tools()
        tool_result = await client.call_tool(
            "search_internal_docs", {"query": "What is data"}
        )

        print("Tools :- \n", tools)
        print("Tool result :- \n", tool_result)
        print("\n\n")
        resource = await client.list_resources()
        print("Resources :-\n", resource)
        resource_result = await client.read_resource("docs://company/about")
        print("Resource result\n", resource_result)
        print("\n\n")
        prompt_template = await client.list_prompts()
        print(prompt_template)
        prompt_result = await client.get_prompt(
            "summarize_document", {"document": "provided documents"}
        )
        print("Prompt result \n", prompt_result)


if __name__ == "__main__":
    anyio.run(main)
