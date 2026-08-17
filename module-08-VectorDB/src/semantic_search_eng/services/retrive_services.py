from semantic_search_eng.user_data.data_manager import (
    DataManager,
)
from semantic_search_eng.user_data.data_processor import (
    DataProcessor,
)
from semantic_search_eng.db.retrive_db import RetriveDBManager

from semantic_search_eng.retrival.retriver_manager import (
    RetriverManager,
)
from semantic_search_eng.models import RetriveRequest, RetriveResponse


class RetriveServiceManager:
    def __init__(self):
        self.data_manager = DataManager()
        self.data_processor = DataProcessor()
        self.retrive_manager = RetriverManager()
        self.retrive_db_manager = RetriveDBManager()

    async def retrive_chunks(self, request: RetriveRequest):
        results = await self.retrive_manager.retrieve(
            tenant_id=request.tenant_id,
            query=request.query,
            top_k=request.top_k,
        )

        return RetriveResponse(
            tenant_id=request.tenant_id,
            query=request.query,
            top_k=request.top_k,
            results=results,
        )
