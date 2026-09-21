"""Lazy imports for the sibling Module 8 and Module 9 applications."""

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _add_source(path: Path) -> None:
    source = str(path)
    if source not in sys.path:
        sys.path.insert(0, source)


def module9_services():
    _add_source(REPOSITORY_ROOT / "module-09-rag" / "src")
    from rag_app.models import RAGRequest
    from rag_app.features.rag_chat import RAGChat

    return RAGChat, RAGRequest


def module8_services():
    _add_source(REPOSITORY_ROOT / "module-08-VectorDB" / "src")
    from semantic_search_eng.models import RetriveRequest
    from semantic_search_eng.services.retrive_services import RetriveServiceManager

    return RetriveServiceManager, RetriveRequest
