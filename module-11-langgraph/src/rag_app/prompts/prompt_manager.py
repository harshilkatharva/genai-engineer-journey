from pathlib import Path

from jinja2 import Template
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from rag_app.core.settings import get_settings
from rag_app.models import PromptRequest


class PromptManager:
    def __init__(self):
        self.settings = get_settings()

    def build_rag_prompt(self, request: PromptRequest):
        context = [chunk.chunk_text for chunk in request.chunks]
        prompt_template = Template(
            Path(
                f"src/rag_app/prompts/services/chat/{self.settings.rag_chat_prompt_running_version}"
            ).read_text()
        )
        return prompt_template.render(
            context=context, user_query=request.query
        ), self.settings.rag_chat_prompt_running_version

    def build_query_expansion_prompt(self, query: str):
        prompt_template = Template(Path("src/rag_app/prompts/query/query_expansion.md").read_text())

        return prompt_template.render(query=query)

    def build_query_HyDE_prompt(self, query: str):
        prompt_template = Template(Path("src/rag_app/prompts/query/query_hyde.md").read_text())
        return prompt_template.render(query=query)

    def build_rag_prompt_langchain(self, format_instructions: str = "") -> ChatPromptTemplate:
        prompt_path = Path(
            f"src/rag_app/prompts/services/chat/{self.settings.rag_chat_prompt_running_version}"
        )

        template = prompt_path.read_text(encoding="utf-8")

        return ChatPromptTemplate.from_messages(
            [
                ("system", template),
                MessagesPlaceholder(variable_name="history", optional=True),
                ("human", "{query}"),
            ]
        ).partial(format_instructions=format_instructions)

    def build_classification_prompt(
        self,
        format_instructions: str = "",
    ) -> ChatPromptTemplate:
        prompt_path = Path(
            "src/rag_app/prompts/services/classification/"
            f"{self.settings.rag_classification_prompt_running_version}"
        )

        return ChatPromptTemplate.from_messages(
            [("system", prompt_path.read_text(encoding="utf-8")), ("human", "{text}")]
        ).partial(format_instructions=format_instructions)

    def build_extraction_prompt(
        self,
        format_instructions: str = "",
    ) -> ChatPromptTemplate:
        prompt_path = Path(
            "src/rag_app/prompts/services/extraction/"
            f"{self.settings.rag_extraction_prompt_running_version}"
        )

        return ChatPromptTemplate.from_messages(
            [("system", prompt_path.read_text(encoding="utf-8")), ("human", "{text}")]
        ).partial(format_instructions=format_instructions)
