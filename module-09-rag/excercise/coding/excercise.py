import re
from dataclasses import dataclass
from typing import Any, Callable
from pydantic import BaseModel, Field


# 1


def format_context(chunks):
    """
    Format retrieved chunks into a numbered, source-labeled context block.

    Each chunk should contain:
      - "text": the retrieved content
      - "source": the source/document name
    """
    formatted = []

    for i, chunk in enumerate(chunks, start=1):
        formatted.append(f"[{i}] Source: {chunk['source']}\n{chunk['text']}")

    return "\n\n".join(formatted)


# 2
def hyde_retrieve(query, llm, retriever, top_k=5):
    hypothetical_doc = llm(
        f"Write a short hypothetical document that would answer this question:\n{query}"
    )

    results = retriever(hypothetical_doc, top_k=top_k)

    return {"hypothetical_doc": hypothetical_doc, "results": results}


def direct_retrieve(query, retriever, top_k=5):
    return retriever(query, top_k=top_k)


def compare_retrievals(query, llm, retriever, top_k=5):
    direct_results = direct_retrieve(query, retriever, top_k=top_k)

    # HyDE retrieval
    hyde_output = hyde_retrieve(query, llm, retriever, top_k=top_k)

    hyde_results = hyde_output["results"]

    # Compare document IDs
    direct_ids = {doc["id"] for doc in direct_results}

    hyde_ids = {doc["id"] for doc in hyde_results}

    overlap = direct_ids & hyde_ids

    return {
        "query": query,
        "hypothetical_doc": hyde_output["hypothetical_doc"],
        "direct_results": direct_results,
        "hyde_results": hyde_results,
        "overlap_count": len(overlap),
        "overlap_ids": list(overlap),
        "direct_only": list(direct_ids - hyde_ids),
        "hyde_only": list(hyde_ids - direct_ids),
    }


# 3
def hybrid_scores(vector_scores, keyword_scores, alpha=0.5):
    def min_max(scores):
        minimum = min(scores)
        maximum = max(scores)

        if maximum == minimum:
            return [1.0] * len(scores)

        return [(s - minimum) / (maximum - minimum) for s in scores]

    vector_norm = min_max(vector_scores)
    keyword_norm = min_max(keyword_scores)

    return [alpha * v + (1 - alpha) * k for v, k in zip(vector_norm, keyword_norm)]


# 4
def validate_citations(answer, retrieved_chunks):
    citations = re.findall(r"\[Source\s+(\d+)\]", answer)

    # Source numbers that actually exist
    valid_sources = {chunk["source"] for chunk in retrieved_chunks}

    cited_sources = {int(source) for source in citations}

    invalid_citations = cited_sources - valid_sources

    unused_sources = valid_sources - cited_sources

    return {
        "valid": len(invalid_citations) == 0,
        "cited_sources": sorted(cited_sources),
        "invalid_citations": sorted(invalid_citations),
        "unused_sources": sorted(unused_sources),
    }


# 5


@dataclass
class EvaluationResult:
    query: str
    retrieved: list[Any]
    answer: str
    retrieval_score: float
    generation_score: float


class RAGEvaluationHarness:
    def __init__(
        self,
        retriever: Callable,
        generator: Callable,
        retrieval_evaluator: Callable,
        generation_evaluator: Callable,
    ):
        self.retriever = retriever
        self.generator = generator
        self.retrieval_evaluator = retrieval_evaluator
        self.generation_evaluator = generation_evaluator

    def evaluate_retrieval(self, query, expected_chunks):
        retrieved = self.retriever(query)

        score = self.retrieval_evaluator(retrieved, expected_chunks)

        return retrieved, score

    def evaluate_generation(self, query, retrieved, expected_answer):
        answer = self.generator(query, retrieved)

        score = self.generation_evaluator(answer, expected_answer, retrieved)

        return answer, score

    def evaluate(self, query, expected_chunks, expected_answer):
        """
        Run the complete retrieval + generation evaluation.
        """

        # Part 1: Retrieval
        retrieved, retrieval_score = self.evaluate_retrieval(query, expected_chunks)

        # Part 2: Generation
        answer, generation_score = self.evaluate_generation(query, retrieved, expected_answer)

        return EvaluationResult(
            query=query,
            retrieved=retrieved,
            answer=answer,
            retrieval_score=retrieval_score,
            generation_score=generation_score,
        )


# 6
class QueryDecomposition(BaseModel):
    sub_queries: list[str] = Field(
        description="Independent sub-queries needed to answer the original query."
    )


def decompose_query(query: str, llm) -> QueryDecomposition:
    prompt = f"""
    Decompose the following complex query into independent sub-queries.

    Rules:
    - Each sub-query should address one distinct information need.
    - Make each sub-query self-contained.
    - Do not add information that is not present in the original query.
    - Return only the structured decomposition.

    Format of the answer :
    ['','']

    Query:
    {query}
    """

    # Configure the LLM to return QueryDecomposition.
    structured_llm = llm.with_structured_output(QueryDecomposition)

    return structured_llm.data(prompt)


# 8
def deduplicate_chunks(chunks, threshold=0.8):
    def similarity(text1, text2):
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        return len(words1 & words2) / len(words1 | words2)

    unique_chunks = []

    for chunk in chunks:
        is_duplicate = False

        for existing in unique_chunks:
            if similarity(chunk["text"], existing["text"]) >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_chunks.append(chunk)

    return unique_chunks


# 9
def rerank_with_fallback(query, chunks, reranker):
    # Preserve the original vector-search ordering
    fallback = sorted(chunks, key=lambda chunk: chunk["vector_score"], reverse=True)

    try:
        reranked = reranker(query, chunks)

        # If reranker returns nothing/invalid output,
        # use the fallback ordering.
        if not reranked:
            return fallback

        return reranked

    except Exception:
        # Re-ranking failure should not fail the request.
        return fallback
