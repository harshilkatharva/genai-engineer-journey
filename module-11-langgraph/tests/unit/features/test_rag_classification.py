from unittest.mock import MagicMock

import pytest
from langchain_core.runnables import RunnableLambda

from rag_app.features.rag_classification import RAGClassification
from rag_app.models import RAGClassificationResponse


@pytest.mark.asyncio
async def test_classify_uses_text_only_and_returns_schema():
    prompt_manager = MagicMock()
    prompt_manager.build_classification_prompt.side_effect = lambda format_instructions: (
        RunnableLambda(lambda value: (f"{value['text']}\n{format_instructions}"))
    )
    llm = RunnableLambda(
        lambda _: '{"category":"billing","confidence":0.9,"reason":"The text is about an invoice."}'
    )

    classification = RAGClassification.__new__(RAGClassification)
    classification.prompt_manager = prompt_manager
    classification.llm_manager = MagicMock()
    classification.llm_manager.get_chat_model.return_value = llm
    classification.chain = classification._build_chain()

    result = await classification.classify("The invoice total is incorrect.")

    assert result == RAGClassificationResponse(
        category="billing",
        confidence=0.9,
        reason="The text is about an invoice.",
    )
    prompt_manager.build_classification_prompt.assert_called_once()
    assert (
        "category"
        in prompt_manager.build_classification_prompt.call_args.kwargs["format_instructions"]
    )
