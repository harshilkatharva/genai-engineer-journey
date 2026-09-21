from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.prompts import prompt

mcp = MCPServer("PromptServer")


@prompt(
    name="rag_answer",
    description="Generate a RAG question-answering prompt.",
)
def rag_answer(question: str, context: str):
    return f"""
You are a helpful assistant.

Answer the user's question using only the provided context.

Context:
{context}

Question:
{question}

If the answer is not present in the context, say:
"I don't know based on the provided context."
"""


if __name__ == "__main__":
    mcp.run(transport="stdio")
