from rag_app.models import (
    LLMManagerRequest,
    PromptRequest,
    QueryManagerRequest,
    RAGRequest,
    RAGResposne,
    RetriveRequest,
)
from rag_app.prompts.prompt_manager import PromptManager
from rag_app.query.query_manager import QueryManager
from rag_app.retrieval.retriver_manager import RetriverManager
from rag_app.services.llm_services import LLMServicemanager


class RAGChat:
    def __init__(self):
        self.query_manager = QueryManager()
        self.retriver_manager = RetriverManager()
        self.prompt_manager = PromptManager()
        self.llm_manager = LLMServicemanager()

    async def get_answer(self, request: RAGRequest) -> RAGResposne:
        queries = await self.query_manager.get_queries(
            request=QueryManagerRequest(query=request.query)
        )
        context = await self.retriver_manager.retrieve(
            request=RetriveRequest(tenant_id=request.tenant_id, queries=queries.queries)
        )
        prompt = self.prompt_manager.build_rag_prompt(
            request=PromptRequest(query=request.query, chunks=context.results)
        )
        answer = await self.llm_manager.complete(request=LLMManagerRequest(prompt=prompt))

        return RAGResposne(text=answer.text)
