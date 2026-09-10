from langchain_core.output_parsers import PydanticOutputParser

from rag_app.models import RAGExtractionResponse
from rag_app.prompts.prompt_manager import PromptManager
from rag_app.services.llm_services import LLMServicemanager


class RAGExtraction:
    def __init__(self):
        self.prompt_manager = PromptManager()
        self.llm_manager = LLMServicemanager()
        self.chain = self._build_chain()

    def _build_chain(self):
        parser = PydanticOutputParser(pydantic_object=RAGExtractionResponse)
        prompt = self.prompt_manager.build_extraction_prompt(
            format_instructions=parser.get_format_instructions()
        )
        llm = self.llm_manager.get_chat_model()

        return prompt | llm | parser

    async def extract(self, text: str) -> RAGExtractionResponse:
        return await self.chain.ainvoke({"text": text})
