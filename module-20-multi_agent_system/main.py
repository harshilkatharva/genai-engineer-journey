"""Interactive CLI entrypoint for Parallel Research Team (Module 20).

Takes configuration directly from the central configuration file
(parallel_research_team.core.config.ResearchTeamConfig), with optional CLI overrides.
Executes multi-agent research runs on custom topics, inspects real-time progress,
and persists checkpoints and telemetry to disk.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from genai_core.embedding.local_provider import LocalEmbeddingProvider
from genai_core.models import DocumentChunk
from genai_core.retrieval.in_memory_store import InMemoryVectorStore
from genai_core.services.embedding_service import EmbeddingService
from genai_core.services.llm_services import LLMService
from genai_core.services.retrieval_service import RetrievalService

from parallel_research_team.core.config import ResearchTeamConfig, get_research_team_config
from parallel_research_team.models.checkpoint import RunStatus
from parallel_research_team.models.decomposition import EffortLevel
from parallel_research_team.orchestrator import ParallelResearchTeam
from parallel_research_team.storage.file_store import FileCheckpointStore
from parallel_research_team.workers.tools import create_retrieval_service

# Load environment variables (.env)
load_dotenv()

DEFAULT_FIXTURES_PATH = (
    Path(__file__).resolve().parent / "tests" / "fixtures" / "research_chunks.json"
)


def _setup_retrieval_service(
    config: ResearchTeamConfig, fixtures_path: Path
) -> RetrievalService | None:
    """Set up retrieval service respecting config.retrieval_mode ('in_memory' or 'postgres')."""
    try:
        # 1. PostgreSQL retrieval mode configured in config
        if config.retrieval_mode == "postgres" and config.postgres_connection_string:
            emb_provider = LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
            emb_service = EmbeddingService(provider=emb_provider)
            print(f"✔ Using PostgreSQL retrieval mode ({config.postgres_connection_string}).")
            return create_retrieval_service(config=config, embedding_service=emb_service)

        # 2. In-Memory retrieval mode using real handbook chunks fixture if present
        if fixtures_path.exists():
            content = fixtures_path.read_text(encoding="utf-8")
            raw_chunks: list[dict[str, Any]] = json.loads(content)[:100]

            document_chunks = [
                DocumentChunk(
                    id=c["chunk_id"],
                    content=c["chunk_text"],
                    embedding=json.loads(c["embedding"])
                    if isinstance(c["embedding"], str)
                    else c.get("embedding", []),
                    metadata=c.get("metadata") or {},
                    source_id=c.get("metadata", {}).get("document_name", "production_handbook"),
                )
                for c in raw_chunks
            ]
            store = InMemoryVectorStore(dimensions=384, chunks=document_chunks)
            emb_provider = LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
            emb_service = EmbeddingService(provider=emb_provider)
            print(
                f"✔ Loaded in-memory knowledge base with {len(document_chunks)} real handbook chunks."
            )
            return RetrievalService(store=store, embedding_service=emb_service)

        print(
            f"[Notice] Fixtures file not found at {fixtures_path}. Initializing default retrieval."
        )
        return create_retrieval_service(config=config)
    except Exception as exc:  # noqa: BLE001
        print(f"[Notice] Could not initialize retrieval service: {exc}. Using web search only.")
        return None


async def run_research(
    topic: str,
    effort: str | None = None,
    max_workers: int | None = None,
    concurrency: int | None = None,
    enable_semantic_verification: bool | None = None,
    resume_run_id: str | None = None,
) -> None:
    """Execute research orchestrator reading settings from central config."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[Error] GOOGLE_API_KEY environment variable is not set.")
        print("Please export GOOGLE_API_KEY='your-key' or set it in .env before running.")
        sys.exit(1)

    # 1. Load base configuration directly from central config file
    base_config = get_research_team_config()

    # 2. Apply explicit CLI overrides only if provided
    overrides: dict[str, Any] = {}
    if max_workers is not None:
        overrides["max_workers"] = max_workers
    if concurrency is not None:
        overrides["max_concurrency"] = concurrency
    if enable_semantic_verification is not None:
        overrides["enable_semantic_verification"] = enable_semantic_verification

    config = base_config.model_copy(update=overrides) if overrides else base_config

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    llm_service = LLMService(
        default_provider=config.default_provider,
        default_model=config.default_model,
    )

    print("\n" + "=" * 80)
    print("PARALLEL RESEARCH TEAM — MULTI-AGENT RUNNER")
    print("=" * 80)
    print("• Config Source:       parallel_research_team.core.config.ResearchTeamConfig")
    print(f"• LLM Provider/Model:  {config.default_provider} / {config.default_model}")
    print(f"• Judge Provider/Model:{config.judge_provider} / {config.judge_model}")
    print(f"• Checkpoint Storage:  {checkpoint_dir.resolve()}")
    print(f"• Max Workers:         {config.max_workers}")
    print(f"• Max Concurrency:     {config.max_concurrency}")
    print(f"• Per-Worker Max Steps:{config.per_worker_max_steps}")
    print(f"• Per-Worker Timeout:  {config.per_worker_timeout_seconds}s")
    print(f"• Retrieval Mode:      {config.retrieval_mode}")
    print(f"• Gate 2 Verification: {config.enable_semantic_verification}")
    print("=" * 80)

    # Initialize retrieval service according to config
    retrieval_service = _setup_retrieval_service(config, DEFAULT_FIXTURES_PATH)

    team = ParallelResearchTeam(
        config=config,
        llm_service=llm_service,
        retrieval_service=retrieval_service,
    )

    # Determine effort mode: CLI argument > config default
    effort_mode_str = effort or config.default_effort_level
    effort_override: EffortLevel | None = None
    if effort_mode_str.lower() == "direct":
        effort_override = EffortLevel.DIRECT
    elif effort_mode_str.lower() == "parallel":
        effort_override = EffortLevel.PARALLEL_DECOMPOSED

    if resume_run_id:
        print(f"\n[Resuming Run] Resuming existing checkpoint '{resume_run_id}'...")
        result = await team.resume(run_id=resume_run_id)
    else:
        print(f"\n[Starting Research] Topic: '{topic}'")
        if effort_override:
            print(f"[Effort Mode Override]: {effort_override.value}")
        else:
            print(
                f"[Effort Mode]: Dynamic Autonomous Classification ({config.default_effort_level})"
            )

        result = await team.run(
            topic=topic,
            effort_override=effort_override,
        )

    # Display Execution Summary
    print("\n" + "=" * 80)
    print("RESEARCH EXECUTION SUMMARY")
    print("=" * 80)
    print(f"• Run ID:           {result.run_id}")
    print(f"• Final Status:     {result.status.value.upper()}")
    print(f"• Total Duration:   {result.metrics.duration_ms / 1000.0:.2f} seconds")
    print(f"• Active Workers:   {result.metrics.total_workers}")
    print(f"• Total Tokens:     {result.metrics.total_tokens:,} tokens")
    print(f"• Total USD Cost:   ${result.metrics.total_cost_usd:.5f}")

    if result.decomposition:
        print("\n" + "-" * 80)
        print("LEAD AGENT DECOMPOSITION PLAN")
        print("-" * 80)
        print(f"• Effort Level: {result.decomposition.effort_level.value}")
        print(f"• Rationale:    {result.decomposition.rationale}")
        print(f"• Briefs Created ({len(result.decomposition.worker_briefs)}):")
        for idx, b in enumerate(result.decomposition.worker_briefs, 1):
            print(f"  {idx}. [{b.brief_id}] {b.title} ({b.brief_type.value})")
            print(f"     Objective:  {b.objective}")
            print(f"     Boundaries: {', '.join(b.task_boundaries)}")

    checkpoint_file = checkpoint_dir / f"checkpoint_{result.run_id}.json"
    print("\n" + "-" * 80)
    print(f"CHECKPOINT ON DISK: {checkpoint_file}")
    if checkpoint_file.exists():
        size_kb = checkpoint_file.stat().st_size / 1024.0
        print(f"✔ Checkpoint file preserved ({size_kb:.1f} KB).")
        print(f"  You can view raw state with: cat {checkpoint_file}")

    if result.report:
        report = result.report
        print("\n" + "=" * 80)
        print(f"FINAL SYNTHESIZED REPORT: {report.title}")
        print("=" * 80)
        print(f"\n[EXECUTIVE SUMMARY]\n{report.executive_summary.content}\n")

        for s in report.sections:
            print(f"\n[{s.title.upper()}]")
            print(s.content)
            if s.claims:
                print("\n  Claims & Grounding:")
                for c in s.claims:
                    cite_ids = [f"[{cit.source_id}]" for cit in c.citations]
                    print(f"  • {c.text} {' '.join(cite_ids)}")

        print(f"\n[CONCLUSION]\n{report.conclusion.content}\n")

        if report.limitations:
            print("[LIMITATIONS]")
            for lim in report.limitations:
                print(f"• {lim}")

        if report.unresolved_limitations:
            print("\n[UNRESOLVED / INCOMPLETE BRIEFS]")
            for unres in report.unresolved_limitations:
                print(f"• {unres}")

        print("\n" + "=" * 80)
        print("CITED AUTHORITATIVE SOURCES")
        print("=" * 80)
        checkpoint_store = FileCheckpointStore(config=config)
        persisted = await checkpoint_store.get_checkpoint(result.run_id)
        if persisted and persisted.authoritative_sources:
            for s_data in persisted.authoritative_sources.values():
                print(f"• [{s_data.get('source_id')}] {s_data.get('title')} ({s_data.get('url')})")
    elif result.status == RunStatus.FAILED:
        print("\n[Execution Failed]")
        print(f"Reason: {result.error_message}")


def main() -> None:
    """CLI Argument Parser and entrypoint."""
    config = get_research_team_config()

    parser = argparse.ArgumentParser(
        description="Parallel Research Team (Module 20) Multi-Agent Research Assistant."
    )
    parser.add_argument(
        "--topic",
        "-t",
        type=str,
        default="Compare vector database indexing architectures (HNSW vs IVFFlat) and evaluation methods in production GenAI",
        help="Research topic or question to investigate.",
    )
    parser.add_argument(
        "--effort",
        "-e",
        type=str,
        choices=["auto", "direct", "parallel"],
        default=None,
        help=f"Effort scaling mode (default from config: '{config.default_effort_level}').",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help=f"Maximum number of parallel research workers (default from config: {config.max_workers}).",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=None,
        help=f"Maximum concurrent workers running in parallel (default from config: {config.max_concurrency}).",
    )
    parser.add_argument(
        "--enable-semantic-judge",
        action="store_true",
        default=None,
        help=f"Enable Module 19 Gate 2 live semantic entailment judge (default from config: {config.enable_semantic_verification}).",
    )
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
        default=None,
        help="Resume an existing run ID from checkpoint file.",
    )

    args = parser.parse_args()

    asyncio.run(
        run_research(
            topic=args.topic,
            effort=args.effort,
            max_workers=args.workers,
            concurrency=args.concurrency,
            enable_semantic_verification=args.enable_semantic_judge,
            resume_run_id=args.resume,
        )
    )


if __name__ == "__main__":
    main()
