"""Versioned prompt template manager powered by Jinja2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jinja2

from ..core.config import ResearchTeamConfig, get_research_team_config

if TYPE_CHECKING:
    from ..models import SourceReference, WorkerBrief


class PromptTemplateManager:
    """Manages loading and rendering versioned prompt templates via Jinja2."""

    def __init__(
        self,
        base_dir: Path | str | None = None,
        default_version: str | None = None,
        config: ResearchTeamConfig | None = None,
    ) -> None:
        cfg = config or get_research_team_config()
        self.default_version = default_version or cfg.prompt_version

        if base_dir is None:
            self.base_dir = Path(__file__).resolve().parent
        else:
            self.base_dir = Path(base_dir).resolve()

        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.base_dir)),
            autoescape=False,
            undefined=jinja2.StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _resolve_template_path(self, relative_path: str, version: str | None = None) -> str:
        """Construct versioned relative template path."""
        ver = version or self.default_version
        clean_path = relative_path.lstrip("/")
        if not clean_path.endswith(".jinja2"):
            clean_path += ".jinja2"
        return f"{ver}/{clean_path}"

    def render_template(
        self,
        relative_path: str,
        version: str | None = None,
        **context: Any,
    ) -> str:
        """Render a versioned Jinja2 template with context."""
        template_path = self._resolve_template_path(relative_path, version)
        try:
            template = self._env.get_template(template_path)
            return template.render(**context).strip()
        except jinja2.TemplateNotFound as exc:
            raise FileNotFoundError(
                f"Prompt template '{template_path}' not found in '{self.base_dir}'"
            ) from exc

    def render_decomposition_prompts(
        self,
        topic: str,
        max_workers: int = 4,
        effort_mode: str = "auto",
        user_preferences: str = "",
        per_worker_max_steps: int = 5,
        per_worker_timeout_seconds: float = 45.0,
        version: str | None = None,
    ) -> tuple[str, str]:
        """Render system and user prompts for research question decomposition."""
        system_prompt = self.render_template(
            "decomposition/system",
            version=version,
            max_workers=max_workers,
            per_worker_max_steps=per_worker_max_steps,
            per_worker_timeout_seconds=per_worker_timeout_seconds,
        )
        user_prompt = self.render_template(
            "decomposition/user",
            version=version,
            topic=topic,
            max_workers=max_workers,
            effort_mode=effort_mode,
            user_preferences=user_preferences,
        )
        return system_prompt, user_prompt

    def render_worker_prompts(
        self,
        brief: WorkerBrief,
        version: str | None = None,
    ) -> tuple[str, str]:
        """Render system and user prompts for an individual research worker."""
        system_prompt = self.render_template(
            "worker/system",
            version=version,
            brief=brief,
        )
        user_prompt = self.render_template(
            "worker/user",
            version=version,
            brief=brief,
        )
        return system_prompt, user_prompt

    def render_synthesis_prompts(
        self,
        topic: str,
        worker_findings: dict[str, str],
        sources: list[SourceReference] | list[dict[str, Any]],
        version: str | None = None,
    ) -> tuple[str, str]:
        """Render system and user prompts for final research report synthesis."""
        system_prompt = self.render_template(
            "synthesis/system",
            version=version,
            topic=topic,
        )
        user_prompt = self.render_template(
            "synthesis/user",
            version=version,
            topic=topic,
            worker_findings=worker_findings,
            sources=sources,
        )
        return system_prompt, user_prompt

    def get_metadata(self, version: str | None = None) -> dict[str, Any]:
        """Load metadata.json for the specified prompt version."""
        ver = version or self.default_version
        meta_path = self.base_dir / ver / "metadata.json"
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def list_versions(self) -> list[str]:
        """List all available prompt version directories."""
        versions: list[str] = []
        for p in self.base_dir.iterdir():
            if p.is_dir() and (p / "metadata.json").exists():
                versions.append(p.name)
        versions.sort()
        return versions
