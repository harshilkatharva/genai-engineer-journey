from uuid import UUID


from .keyword_search import KeywordSearch
from .vector_search import VectorSearch


class HybridSearch:
    def __init__(self):
        self.keyword_search = KeywordSearch()
        self.vector_search = VectorSearch()

    def retrive(self, tenant_id: UUID, queries: list[str], top_k_candidates: int):
        pass

    def _retrive_vectors(self):
        pass

    def _retrive_keyword(self):
        pass
