from ..schemas.rag import QueryRoute


class QueryRouter:
    """Deterministic router that can later be replaced by an LLM classifier."""

    _synthesis_terms = (
        "summarize",
        "summary",
        "overview",
        "compare",
        "contrast",
        "overall",
        "themes",
        "trends",
        "across",
        "in general",
    )

    def classify(self, query: str) -> QueryRoute:
        normalized = query.lower()
        return "summary" if any(term in normalized for term in self._synthesis_terms) else "vector"
