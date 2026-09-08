from unittest.mock import MagicMock

import pytest
from langchain_core.runnables import RunnableLambda

from rag_app.features.rag_extraction import RAGExtraction
from rag_app.models import RAGExtractionResponse


@pytest.mark.asyncio
async def test_extract_uses_text_only_and_returns_schema():
    prompt_manager = MagicMock()
    prompt_manager.build_extraction_prompt.side_effect = lambda format_instructions: (
        RunnableLambda(lambda value: f"{value['text']}\n{format_instructions}")
    )
    llm = RunnableLambda(lambda _: '{"fields":{"invoice_id":"INV-42","total":125}}')

    extraction = RAGExtraction.__new__(RAGExtraction)
    extraction.prompt_manager = prompt_manager
    extraction.llm_manager = MagicMock()
    extraction.llm_manager.get_chat_model.return_value = llm
    extraction.chain = extraction._build_chain()

    result = await extraction.extract("Invoice INV-42 has a total of 125.")

    assert result == RAGExtractionResponse(fields={"invoice_id": "INV-42", "total": 125})
    prompt_manager.build_extraction_prompt.assert_called_once()
    assert (
        "fields" in prompt_manager.build_extraction_prompt.call_args.kwargs["format_instructions"]
    )
