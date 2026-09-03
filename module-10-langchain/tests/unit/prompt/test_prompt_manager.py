from unittest.mock import MagicMock, patch

from rag_app.models.prompt.prompt_request import PromptRequest
from rag_app.models.retrive.retrive_response import RetriveResult
from rag_app.prompts.prompt_manager import PromptManager


def test_build_rag_prompt_renders_query_and_chunk_text(tmp_path):
    prompt_file = tmp_path / "rag_v1.md"

    prompt_file.write_text(
        """You are a helpful RAG assistant.

Context:
{% for chunk in context %}
- {{ chunk }}
{% endfor %}

User Question:
{{ user_query }}
"""
    )

    chunks = [
        RetriveResult(
            chunk_id="test_chunk_01",
            chunk_text="Refunds are allowed within 30 days.",
            similarity_score=0.95,
        ),
        RetriveResult(
            chunk_id="test_chunk_02",
            chunk_text="A receipt is required for refunds.",
            similarity_score=0.87,
        ),
    ]

    request = PromptRequest(
        query="What is the refund policy?",
        chunks=chunks,
    )

    settings = MagicMock()
    settings.rag_prompt_running_version = "rag_v1.md"

    with patch(
        "rag_app.prompts.prompt_manager.get_settings",
        return_value=settings,
    ):
        manager = PromptManager()

        # Replace the path used by the manager with our temporary file.
        with patch(
            "rag_app.prompts.prompt_manager.Path",
            return_value=prompt_file,
        ):
            result, prompt_version = manager.build_rag_prompt(request)
            print(result)

    assert "What is the refund policy?" in result
    assert "Refunds are allowed within 30 days." in result
    assert "A receipt is required for refunds." in result
    assert prompt_version == "rag_v1.md"

    # Similarity scores should NOT appear in the prompt.
    assert "0.95" not in result
    assert "0.87" not in result
