from dataclasses import dataclass
from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode

from ..core.config import Settings


@dataclass(frozen=True)
class IngestedCorpus:
    source_dir: Path
    documents: list[Document]
    nodes: list[BaseNode]


class DocumentLoader:
    """Loads supported files and creates consistent chunks for every index."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def load(self, data_dir: str | Path | None = None) -> IngestedCorpus:
        source_dir = Path(data_dir or self.settings.data_dir).expanduser().resolve()
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Data directory does not exist: {source_dir}")
        documents = SimpleDirectoryReader(
            input_dir=source_dir,
            recursive=True,
            exclude_hidden=True,
            filename_as_id=True,
        ).load_data()
        if not documents:
            raise ValueError(f"No supported documents found in {source_dir}")
        nodes = SentenceSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        ).get_nodes_from_documents(documents, show_progress=False)
        return IngestedCorpus(source_dir, documents, nodes)
