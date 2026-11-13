"""Prompt template manager powered by Jinja2 for versioned, isolated LLM prompt management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jinja2

from self_improving_report_writer.core.sanitization import (
    escape_xml_context,
    format_untrusted_source_block,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from self_improving_report_writer.models.critique import Critique
    from self_improving_report_writer.models.report import Report, SourceReference
    from self_improving_report_writer.models.rubric import RubricConfig
    from self_improving_report_writer.models.verification import DeterministicVerificationReport


class PromptTemplateManager:
    """Manages versioned prompt templates loaded and rendered via Jinja2."""

    def __init__(
        self,
        template_dir: Path | str | None = None,
        default_version: str = "v1",
    ) -> None:
        """Initialize the template manager.

        Args:
            template_dir: Base directory containing versioned template folders. Defaults to
                          'prompt_templates/' adjacent to the package.
            default_version: Default template version tag (e.g. 'v1').
        """
        if template_dir is None:
            self.template_dir = Path(__file__).resolve().parent.parent / "prompt_templates"
        else:
            self.template_dir = Path(template_dir).resolve()

        self.default_version = default_version
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir)),
            autoescape=False,
            undefined=jinja2.StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._env.filters["escape_xml"] = escape_xml_context

    @property
    def jinja_env(self) -> jinja2.Environment:
        """Return the underlying Jinja2 Environment."""
        return self._env

    def get_template_path(self, relative_path: str, version: str | None = None) -> str:
        """Construct versioned relative template path."""
        ver = version or self.default_version
        clean_rel = relative_path.lstrip("/")
        if not clean_rel.endswith(".jinja2"):
            clean_rel += ".jinja2"
        return f"{ver}/{clean_rel}"

    def render_template(
        self,
        template_name: str,
        version: str | None = None,
        **context: Any,
    ) -> str:
        """Render a versioned Jinja2 prompt template with the provided context dictionary."""
        template_path = self.get_template_path(template_name, version)
        try:
            tpl = self._env.get_template(template_path)
            return tpl.render(**context).strip()
        except jinja2.TemplateNotFound as e:
            raise FileNotFoundError(
                f"Prompt template '{template_path}' not found in {self.template_dir}"
            ) from e

    def get_system_prompt(
        self,
        domain: str,
        prompt_name: str = "system",
        version: str | None = None,
    ) -> str:
        """Load and return a system prompt for a specific domain (generation, revision, judging, verification)."""
        template_path = f"{domain}/{prompt_name}"
        return self.render_template(template_path, version=version)

    def list_templates(self, version: str | None = None) -> list[str]:
        """List all available template names under a given version."""
        ver = version or self.default_version
        ver_dir = self.template_dir / ver
        if not ver_dir.exists():
            return []
        templates: list[str] = []
        for file in sorted(ver_dir.rglob("*.jinja2")):
            rel = file.relative_to(ver_dir).as_posix()
            templates.append(rel)
        return templates

    def get_version_metadata(self, version: str | None = None) -> dict[str, Any]:
        """Return version metadata if metadata.json exists for the specified version."""
        ver = version or self.default_version
        meta_file = self.template_dir / ver / "metadata.json"
        if meta_file.exists():
            try:
                return json.loads(meta_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {"version": ver}
        return {"version": ver}

    def format_source_contexts(self, sources: Sequence[SourceReference]) -> str:
        """Format source chunks inside strict XML-like data boundaries."""
        if not sources:
            return "No source context provided."

        formatted_chunks: list[str] = []
        for s in sources:
            formatted_chunks.append(
                format_untrusted_source_block(
                    s.content,
                    id=str(s.chunk_id),
                    source_id=str(s.source_id),
                    title=str(s.title),
                )
            )
        return "\n\n".join(formatted_chunks)

    def render_report_generation_prompt(
        self,
        topic: str,
        sources: Sequence[SourceReference],
        guidelines: list[str] | None = None,
        version: str | None = None,
    ) -> str:
        """Render complete user prompt for report generation."""
        sources_text = self.format_source_contexts(sources)
        return self.render_template(
            "generation/user",
            version=version,
            topic=topic,
            guidelines=guidelines,
            sources_text=sources_text,
        )

    def render_revision_prompt(
        self,
        current_report: Report,
        sources: list[SourceReference],
        critique: Critique,
        version: str | None = None,
    ) -> str:
        """Render user prompt for report revision with critique feedback."""
        sources_text = self.format_source_contexts(sources)

        crit_lines = []
        for issue in critique.issues:
            loc = ""
            if issue.affected_section_id:
                loc += f" [Section: {issue.affected_section_id}]"
            if issue.affected_claim_id:
                loc += f" [Claim: {issue.affected_claim_id}]"

            crit_lines.append(
                f"- **[{issue.severity.upper()}] ({issue.category})**{loc}:\n"
                f"  * Deficiency: {issue.explanation}\n"
                f"  * Suggested Action: {issue.suggested_correction}"
            )
        critique_block = "\n".join(crit_lines) if crit_lines else "No specific issues identified."

        report_dict = current_report.model_dump(mode="json")
        report_json = json.dumps(report_dict, indent=2)

        return self.render_template(
            "revision/user",
            version=version,
            topic=current_report.topic,
            round_number=critique.round_number,
            critique_block=critique_block,
            sources_text=sources_text,
            current_version=current_report.version,
            report_json=report_json,
            target_version=current_report.version + 1,
        )

    def render_rubric_judge_prompt(
        self,
        report: Report,
        rubric_config: RubricConfig,
        sources: dict[str, SourceReference],
        deterministic_report: DeterministicVerificationReport | None = None,
        version: str | None = None,
    ) -> str:
        """Render user prompt for rubric judge evaluation."""
        criteria_lines = []
        for c in rubric_config.criteria:
            criteria_lines.append(
                f"- **{c.name}** (weight: {c.weight:.2f}, min_pass_score: {c.min_pass_score:.2f}): {c.description}"
            )
        criteria_block = "\n".join(criteria_lines)

        report_dict = report.model_dump(mode="json")
        report_json = json.dumps(report_dict, indent=2)

        source_blocks = [
            format_untrusted_source_block(
                s_ref.content,
                id=str(s_id),
                source_id=str(s_ref.source_id),
                title=str(s_ref.title),
            )
            for s_id, s_ref in sorted(sources.items())
        ]
        sources_text = "\n\n".join(source_blocks) if source_blocks else "No sources provided."

        if deterministic_report:
            det_status = (
                f"Gate 1 Deterministic Verification Status: {'PASSED' if deterministic_report.passed else 'FAILED'}\n"
                f"- Schema Valid: {deterministic_report.schema_valid}\n"
                f"- Citations Valid: {deterministic_report.citations_valid}\n"
                f"- Passages Matched: {deterministic_report.passages_matched}\n"
                f"- Critical Issues: {deterministic_report.critical_count}, Major: {deterministic_report.major_count}, Minor: {deterministic_report.minor_count}\n"
            )
            if deterministic_report.issues:
                issues_summary = "\n".join(
                    f"  * [{issue.severity.upper()}] ({issue.category}): {issue.message}"
                    for issue in deterministic_report.issues[:10]
                )
                det_status += f"Detected Gate 1 Issues:\n{issues_summary}\n"
        else:
            det_status = "Gate 1 Deterministic Verification Status: Not provided."

        return self.render_template(
            "judging/user",
            version=version,
            criteria_block=criteria_block,
            det_status=det_status,
            topic=report.topic,
            sources_text=sources_text,
            report_json=report_json,
            num_criteria=len(rubric_config.criteria),
        )

    def render_claim_entailment_prompt(
        self,
        claim_id: str,
        claim_text: str,
        cited_chunks: list[tuple[str, str]],
        version: str | None = None,
    ) -> str:
        """Render user prompt for single claim entailment verification."""
        chunk_blocks = [
            format_untrusted_source_block(content, chunk_id=str(c_id))
            for c_id, content in cited_chunks
        ]
        evidence_text = "\n\n".join(chunk_blocks) if chunk_blocks else "NO CITED CHUNKS PROVIDED."

        return self.render_template(
            "verification/claim_entailment_user",
            version=version,
            claim_id=claim_id,
            claim_text=claim_text,
            evidence_text=evidence_text,
        )

    def render_batch_claim_entailment_prompt(
        self,
        section_id: str,
        section_title: str,
        claims: list[tuple[str, str, list[str]]],
        cited_chunks: list[tuple[str, str]],
        version: str | None = None,
    ) -> str:
        """Render user prompt for batch claim entailment verification."""
        chunk_blocks = [
            format_untrusted_source_block(content, chunk_id=str(c_id))
            for c_id, content in cited_chunks
        ]
        evidence_text = "\n\n".join(chunk_blocks) if chunk_blocks else "NO CITED CHUNKS PROVIDED."

        claim_blocks = []
        for c_id, text, c_chunks in claims:
            chunks_str = ", ".join(c_chunks) if c_chunks else "None"
            claim_blocks.append(
                f'- Claim ID: {c_id}\n  Statement: "{text}"\n  Referenced Chunk IDs: [{chunks_str}]'
            )
        claims_text = "\n\n".join(claim_blocks) if claim_blocks else "NO CLAIMS TO EVALUATE."

        return self.render_template(
            "verification/batch_claim_entailment_user",
            version=version,
            section_id=section_id,
            section_title=section_title,
            claims_text=claims_text,
            evidence_text=evidence_text,
        )

    def render_prose_consistency_prompt(
        self,
        section_id: str,
        section_title: str,
        prose_content: str,
        structured_claims: list[str],
        version: str | None = None,
    ) -> str:
        """Render user prompt for prose consistency verification."""
        claims_list = (
            "\n".join(f"- {c}" for c in structured_claims)
            if structured_claims
            else "No structured claims defined for this section."
        )

        return self.render_template(
            "verification/prose_consistency_user",
            version=version,
            section_id=section_id,
            section_title=section_title,
            prose_content=prose_content,
            claims_list=claims_list,
        )


_DEFAULT_MANAGER: PromptTemplateManager | None = None


def get_prompt_manager() -> PromptTemplateManager:
    """Return shared default PromptTemplateManager singleton instance."""
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        _DEFAULT_MANAGER = PromptTemplateManager()
    return _DEFAULT_MANAGER
