from pathlib import Path

from ..core.config import Settings
from ..indexes.registry import IndexFactory, IndexRegistry
from ..ingestion.loader import DocumentLoader, IngestedCorpus
from ..models.providers import ModelProvider
from ..retrieval.engines import RetrievalEngineFactory
from ..routing.classifier import QueryRouter
from ..schemas.rag import IngestResponse, QueryResponse, QueryRoute, SourceReference


class RAGService:
    """Facade coordinating ingestion, indexes, routing, and retrieval."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.loader = DocumentLoader(settings)
        self.index_factory = IndexFactory(settings)
        self.router = QueryRouter()
        self.indexes: IndexRegistry | None = None
        self.corpus: IngestedCorpus | None = None
        self._document_count = 0
        self._node_count = 0
        self._restore_persisted_indexes()

    @property
    def ready(self) -> bool:
        return self.indexes is not None

    @property
    def document_count(self) -> int:
        return self._document_count

    def ingest(self, data_dir: str | Path | None = None) -> IngestResponse:
        ModelProvider(self.settings).configure()
        self.corpus = self.loader.load(data_dir)
        self.indexes = self.index_factory.build(self.corpus.nodes)
        self._document_count = len(self.corpus.documents)
        self._node_count = len(self.corpus.nodes)
        self.index_factory.persist_manifest(
            source_dir=self.corpus.source_dir,
            documents=len(self.corpus.documents),
            nodes=len(self.corpus.nodes),
        )
        return IngestResponse(
            source_dir=str(self.corpus.source_dir),
            documents=len(self.corpus.documents),
            nodes=len(self.corpus.nodes),
        )

    def query(self, query: str, route: QueryRoute | None = None) -> QueryResponse:
        if self.indexes is None:
            raise RuntimeError("The indexes are not ready. Run POST /ingest first.")
        selected_route = route or self.router.classify(query)
        engines = RetrievalEngineFactory(self.settings, self.indexes)
        response = (engines.vector() if selected_route == "vector" else engines.summary()).query(
            query
        )
        sources = [
            SourceReference(
                source=str(
                    node.node.metadata.get("file_name")
                    or node.node.metadata.get("file_path")
                    or "unknown"
                ),
                snippet=node.node.get_content()[:300],
            )
            for node in (getattr(response, "source_nodes", []) or [])
        ]
        return QueryResponse(answer=str(response), route=selected_route, sources=sources)

    def _restore_persisted_indexes(self) -> None:
        if not self.settings.postgres_url or not self.index_factory.manifest_path.is_file():
            return
        ModelProvider(self.settings).configure()
        restored = self.index_factory.restore()
        if restored is None:
            return
        self.indexes, metadata = restored
        self._document_count = int(metadata["documents"])
        self._node_count = int(metadata["nodes"])
