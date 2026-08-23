from rag_app.core import get_settings
from rag_app.models import QueryManagerRequest, QueryResponse
from rag_app.query.techniques.query_expansion import QueryExpansion
from rag_app.query.techniques.query_hyde import QueryHyDE
from rag_app.observability.logger import logger


class QueryManager:
    def __init__(self):
        self.settings = get_settings()
        self.types = {"query_expansion": QueryExpansion(), "query_HyDE": QueryHyDE()}

    async def get_queries(self, request: QueryManagerRequest) -> QueryResponse:
        technique = (
            self.settings.default_query_strategy if request.technique is None else request.technique
        )

        if technique is None:
            logger.info(
                "Query technique is None", event="query_technique_selected", component="query"
            )

            return QueryResponse(queries=[request.query])
        else:
            queries = await self.types[technique].process_query(request.query)

            return QueryResponse(queries=queries)
