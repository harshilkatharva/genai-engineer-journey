from dataclasses import dataclass

from llama_index.core import Settings as LlamaSettings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

from ..core.config import Settings


@dataclass(frozen=True)
class ModelProvider:
    """Configures the embedding and generation models used by all indexes."""

    settings: Settings

    def configure(self) -> None:
        LlamaSettings.embed_model = HuggingFaceEmbedding(model_name=self.settings.embedding_model)
        LlamaSettings.llm = GoogleGenAI(model=self.settings.llm_model)
