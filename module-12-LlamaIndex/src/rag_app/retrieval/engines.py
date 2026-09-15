from llama_index.core import get_response_synthesizer
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import BaseQueryEngine, RetrieverQueryEngine

from ..core.config import Settings
from ..indexes.registry import IndexRegistry


class RetrievalEngineFactory:
    """Creates query engines with route-specific retrieval behavior."""

    def __init__(self, settings: Settings, indexes: IndexRegistry) -> None:
        self.settings = settings
        self.indexes = indexes

    def vector(self) -> BaseQueryEngine:
        reranker = SentenceTransformerRerank(
            model=self.settings.reranker_model,
            top_n=self.settings.reranker_top_n,
        )
        retriever = self.indexes.vector.as_retriever(
            similarity_top_k=self.settings.similarity_top_k
        )
        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=get_response_synthesizer(),
            node_postprocessors=[reranker],
        )

    def summary(self) -> BaseQueryEngine:
        return self.indexes.summary.as_query_engine(response_mode="tree_summarize")
