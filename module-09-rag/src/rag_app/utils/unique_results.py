from rag_app.models import RetriveResult


def fetch_unique_results(results: list[RetriveResult], top_k_candidates: int):
    unique_results: dict[str, RetriveResult] = {}

    for result in results:
        existing = unique_results.get(result.chunk_id)
        if existing is None or result.similarity_score > existing.similarity_score:
            unique_results[result.chunk_id] = result

    # Sort by highest similarity score first
    final_results = sorted(
        unique_results.values(),
        key=lambda x: x.similarity_score,
        reverse=True,
    )[:top_k_candidates]

    return final_results
