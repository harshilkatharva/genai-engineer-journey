from rag_app.db.retrive_db import RetriveDBManager
from rag_app.models import RetriveRequest, RetriveResponse
from rag_app.retrieval.retriver_manager import (
    RetriverManager,
)
from rag_app.user_data.data_manager import (
    DataManager,
)
from rag_app.user_data.data_processor import (
    DataProcessor,
)


class RetriveServiceManager:
    def __init__(self):
        self.data_manager = DataManager()
        self.data_processor = DataProcessor()
        self.retrive_manager = RetriverManager()
        self.retrive_db_manager = RetriveDBManager()

    async def retrive_chunks(self, request: RetriveRequest):
        results = await self.retrive_manager.retrieve(request=request)

        return RetriveResponse(
            tenant_id=request.tenant_id,
            queries=request.queries,
            results=results.results,
        )
