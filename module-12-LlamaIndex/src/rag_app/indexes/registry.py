import json
from dataclasses import dataclass
from pathlib import Path

from llama_index.core import (
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.postgres import PGVectorStore

from ..core.config import Settings


@dataclass
class IndexRegistry:
    vector: VectorStoreIndex
    summary: SummaryIndex


class IndexFactory:
    """Builds the fact-retrieval and synthesis indexes from the same nodes."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def manifest_path(self) -> Path:
        return self.settings.index_storage_dir / "manifest.json"

    def _vector_store(self) -> PGVectorStore:
        if not self.settings.postgres_url:
            raise RuntimeError("POSTGRES_URL must be configured before loading indexes.")
        return PGVectorStore(
            connection_string=self.settings.postgres_url,
            table_name=self.settings.vector_table_name,
            schema_name=self.settings.vector_schema_name,
            embed_dim=self.settings.embedding_dimension,
        )

    def build(self, nodes: list[BaseNode]) -> IndexRegistry:
        vector_store = self._vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        summary_storage = StorageContext.from_defaults()
        summary_index = SummaryIndex(nodes, storage_context=summary_storage, show_progress=False)
        self.settings.index_storage_dir.mkdir(parents=True, exist_ok=True)
        summary_storage.persist(persist_dir=self.settings.index_storage_dir)
        return IndexRegistry(
            vector=VectorStoreIndex(nodes, storage_context=storage_context, show_progress=False),
            summary=summary_index,
        )

    def persist_manifest(self, *, source_dir: Path, documents: int, nodes: int) -> None:
        self.settings.index_storage_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(
                {
                    "source_dir": str(source_dir),
                    "documents": documents,
                    "nodes": nodes,
                    "vector_table_name": self.settings.vector_table_name,
                    "vector_schema_name": self.settings.vector_schema_name,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def restore(self) -> tuple[IndexRegistry, dict[str, int | str]] | None:
        if not self.manifest_path.is_file():
            return None
        try:
            metadata = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            storage_context = StorageContext.from_defaults(
                persist_dir=self.settings.index_storage_dir,
            )
            summary_index = load_index_from_storage(storage_context)
            vector_index = VectorStoreIndex.from_vector_store(self._vector_store())
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            raise RuntimeError(
                f"Persisted index state could not be restored from {self.settings.index_storage_dir}."
            ) from exc
        return IndexRegistry(vector=vector_index, summary=summary_index), metadata
