from unittest.mock import MagicMock, patch

from rag_app.retrieval.engines import RetrievalEngineFactory


def test_vector_engine_builds_retriever_with_reranker() -> None:
    settings = MagicMock(
        reranker_model="local-reranker",
        reranker_top_n=4,
        similarity_top_k=10,
    )
    vector_index = MagicMock()
    retriever = MagicMock()
    vector_index.as_retriever.return_value = retriever
    indexes = MagicMock(vector=vector_index)
    query_engine = MagicMock()

    with (
        patch("rag_app.retrieval.engines.SentenceTransformerRerank") as rerank_cls,
        patch(
            "rag_app.retrieval.engines.RetrieverQueryEngine", return_value=query_engine
        ) as engine_cls,
        patch("rag_app.retrieval.engines.get_response_synthesizer", return_value=MagicMock()),
    ):
        result = RetrievalEngineFactory(settings, indexes).vector()

    rerank_cls.assert_called_once_with(model="local-reranker", top_n=4)
    vector_index.as_retriever.assert_called_once_with(similarity_top_k=10)
    engine_cls.assert_called_once()
    assert result is query_engine


def test_summary_engine_uses_tree_summarize() -> None:
    summary_engine = MagicMock()
    indexes = MagicMock()
    indexes.summary.as_query_engine.return_value = summary_engine

    result = RetrievalEngineFactory(MagicMock(), indexes).summary()

    indexes.summary.as_query_engine.assert_called_once_with(response_mode="tree_summarize")
    assert result is summary_engine
