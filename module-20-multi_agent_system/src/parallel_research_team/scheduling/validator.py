"""Worker output and citation integrity validator."""

from __future__ import annotations

from genai_core.observability import get_logger

from ..models import WorkerBrief, WorkerExecutionResult, WorkerStatus

_logger = get_logger("parallel_research_team.scheduling.validator")


class WorkerOutputValidator:
    """Validates worker execution results, deliverable structures, and cited sources."""

    def validate(
        self,
        result: WorkerExecutionResult,
        brief: WorkerBrief,
    ) -> WorkerExecutionResult:
        """Validate deliverable structure and source citations.

        Returns an updated WorkerExecutionResult with adjusted status or error details if defects
        are detected.
        """
        # If worker already failed or timed out, preserve error and any salvaged sources
        if result.status in (WorkerStatus.FAILED, WorkerStatus.TIMEOUT):
            return result

        # Completed or Partial workers must provide an output
        if result.output is None:
            _logger.warning(
                "Worker for brief '%s' completed without output deliverable", brief.brief_id
            )
            result.status = WorkerStatus.FAILED
            result.error_message = (
                result.error_message
                or f"Worker for brief '{brief.brief_id}' completed without output deliverable."
            )
            return result

        output = result.output

        # 1. Non-empty findings summary validation
        if not output.findings_summary or not output.findings_summary.strip():
            _logger.warning("Worker for brief '%s' has empty findings_summary", brief.brief_id)
            result.status = WorkerStatus.PARTIAL
            err = f"Worker for brief '{brief.brief_id}' produced empty findings summary."
            result.error_message = err
            if err not in output.limitations:
                output.limitations.append(err)
            return result

        # 2. Citation source existence check
        # Build set of known sources collected by this worker
        known_source_keys: set[tuple[str, str]] = set()
        known_chunk_ids: set[str] = set()

        for s in output.sources:
            src_id = str(s.get("source_id", "")).strip()
            chk_id = str(s.get("chunk_id", "")).strip()
            if src_id and chk_id:
                known_source_keys.add((src_id, chk_id))
            if chk_id:
                known_chunk_ids.add(chk_id)

        phantom_citations: list[str] = []
        for claim in output.claims:
            for cite in claim.get("citations", []):
                cite_src = str(cite.get("source_id", "")).strip()
                cite_chk = str(cite.get("chunk_id", "")).strip()

                is_known = (cite_src, cite_chk) in known_source_keys or cite_chk in known_chunk_ids
                if not is_known:
                    phantom_citations.append(f"({cite_src}, {cite_chk})")

        if phantom_citations:
            _logger.warning(
                "Worker for brief '%s' produced citations referencing unknown sources: %s",
                brief.brief_id,
                phantom_citations,
            )
            warn_msg = (
                f"Validation note: Citations {', '.join(phantom_citations)} reference "
                "sources not present in the worker's retrieved source list."
            )
            if warn_msg not in output.limitations:
                output.limitations.append(warn_msg)
            # Downgrade to partial if citations cannot be grounded
            if result.status == WorkerStatus.COMPLETED and not output.sources:
                result.status = WorkerStatus.PARTIAL
                result.error_message = "Output claims cannot be verified: zero sources retrieved."

        # 3. Format required fields check
        required_fields = brief.expected_output_format.required_fields
        if required_fields:
            missing_fields: list[str] = []
            for field in required_fields:
                if field.lower() not in output.findings_summary.lower():
                    missing_fields.append(field)

            if missing_fields:
                _logger.info(
                    "Worker for brief '%s' missing suggested format fields: %s",
                    brief.brief_id,
                    missing_fields,
                )
                fmt_note = f"Output format missing expected elements: {', '.join(missing_fields)}"
                if fmt_note not in output.limitations:
                    output.limitations.append(fmt_note)

        return result
