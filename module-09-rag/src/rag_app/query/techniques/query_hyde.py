from rag_app.models import LLMManagerRequest, QueryHyDEModel
from rag_app.observability.logger import logger
from rag_app.prompts.prompt_manager import PromptManager
from rag_app.services.llm_services import LLMServicemanager


class QueryHyDE:
    def __init__(self):
        self.prompt_manager = PromptManager()
        self.llm_service_manager = LLMServicemanager()

    async def process_query(self, query: str) -> list[str]:
        prompt = self._build_prompt(query=query)
        queries = await self._call_llm(prompt=prompt)
        logger.info(
            "Queries Extracted From LLM",
            event="query_technique_HyDE",
            component="query",
        )
        return queries

    def _build_prompt(self, query: str):
        return self.prompt_manager.build_query_HyDE_prompt(query=query)

    async def _call_llm(self, prompt: str):
        answer = await self.llm_service_manager.complete(
            LLMManagerRequest(prompt=prompt, response_schema=QueryHyDEModel)
        )
        print(answer)
        return answer.data["hypothetical_document"]
