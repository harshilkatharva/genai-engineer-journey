from pathlib import Path

from jinja2 import Template

from rag_app.core.settings import get_settings
from rag_app.models import PromptRequest


class PromptManager:
    def __init__(self):
        self.settings = get_settings()

    def build_rag_prompt(self, request: PromptRequest):
        context = [chunk.chunk_text for chunk in request.chunks]
        prompt_template = Template(
            Path(
                f"src/rag_app/prompts/services/{self.settings.rag_prompt_running_version}"
            ).read_text()
        )
        return prompt_template.render(
            context=context, user_query=request.query
        ), self.settings.rag_prompt_running_version

    def build_query_expansion_prompt(self, query: str):
        prompt_template = Template(Path("src/rag_app/prompts/query/query_expansion.md").read_text())

        return prompt_template.render(query=query)

    def build_query_HyDE_prompt(self, query: str):
        prompt_template = Template(Path("src/rag_app/prompts/query/query_hyde.md").read_text())
        print("HyDE PRompt use")
        return prompt_template.render(query=query)
