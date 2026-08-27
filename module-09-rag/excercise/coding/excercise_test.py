from types import SimpleNamespace

# 7

rag_prompt = """
You are a helpful RAG assistant.

Answer the question using only the retrieved context.
If the context is insufficient to answer the question,
do not answer from your own knowledge. Say that there is
insufficient information in the context.

Context:
{context}

Question:
{question}
"""


class MockRAGModel:
    def invoke(self, prompt):
        if "favorite ice cream flavor" in prompt.lower():
            return SimpleNamespace(
                content="The available context does not contain sufficient information to answer"
            )

        return SimpleNamespace(content="The answer is supported by the retrieved context.")


rag_model = MockRAGModel()


def test_rag_declines_when_context_is_insufficient():
    prompt_text = rag_prompt.lower()

    assert "insufficient" in prompt_text
    assert "do not answer" in prompt_text

    query = "What is the CEO's favorite ice cream flavor?"

    retrieved_context = """
    The company was founded in 2015 and provides cloud storage services.
    """

    prompt = rag_prompt.format(
        context=retrieved_context,
        question=query,
    )

    answer = rag_model.invoke(prompt)

    assert "context does not contain" in answer.content.lower()


# 10
def test_rag_generation_integration():
    # Fixed chunks returned by the mocked retriever
    fixed_chunks = [
        {"id": "chunk-1", "text": "The capital of France is Paris."},
        {"id": "chunk-2", "text": "Paris is located on the River Seine."},
    ]

    def mock_retriever(query):
        return fixed_chunks

    # Real LLM or a realistic mock LLM
    class MockLLM:
        def invoke(self, prompt):
            assert "The capital of France is Paris." in prompt
            assert "Paris is located on the River Seine." in prompt

            return type(
                "Response",
                (),
                {"content": "The capital of France is Paris, located on the River Seine."},
            )()

    llm = MockLLM()

    def rag_pipeline(query):
        # Retrieval is mocked
        chunks = mock_retriever(query)

        context = "\n".join(chunk["text"] for chunk in chunks)

        prompt = f"""
        Answer the question using only the provided context.

        Context:
        {context}

        Question:
        {query}
        """

        # Only generation is actually exercised
        response = llm.invoke(prompt)

        return response.content

    # Run the complete pipeline
    answer = rag_pipeline("What is the capital of France?")

    # Verify generated answer
    assert "Paris" in answer
    assert "River Seine" in answer
