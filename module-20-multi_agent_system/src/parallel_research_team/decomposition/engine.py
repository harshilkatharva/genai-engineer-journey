"""Decomposition engine producing typed worker briefs with dynamic effort scaling."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from ..core.budget import SharedRunBudget
from ..core.config import ResearchTeamConfig, get_research_team_config
from ..exceptions import DecompositionError
from ..models import EffortLevel, ResearchDecomposition
from ..prompts.manager import PromptTemplateManager
from ..telemetry.pricing import PricingCalculator
from .validator import DecompositionValidator

if TYPE_CHECKING:
    from genai_core.services.llm_services import LLMService
    from genai_core.services.memory_service import MemoryService

_logger = logging.getLogger("parallel_research_team.decomposition.engine")


class DecompositionEngine:
    """Orchestrates research question analysis, effort scaling, prompt rendering, and validation."""

    def __init__(
        self,
        llm_service: LLMService | None = None,
        config: ResearchTeamConfig | None = None,
        prompt_manager: PromptTemplateManager | None = None,
        validator: DecompositionValidator | None = None,
        memory_service: MemoryService | None = None,
        pricing_calculator: PricingCalculator | None = None,
    ) -> None:
        self.config = config or get_research_team_config()
        if llm_service is not None:
            self.llm_service = llm_service
        else:
            from genai_core.services.llm_services import LLMService

            self.llm_service = LLMService(
                default_provider=self.config.default_provider,
                default_model=self.config.default_model,
            )
        self.prompt_manager = prompt_manager or PromptTemplateManager(config=self.config)
        self.validator = validator or DecompositionValidator()
        self.memory_service = memory_service
        self.pricing_calculator = pricing_calculator or PricingCalculator()
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0
        self.last_total_tokens: int = 0
        self.last_cost_usd: float = 0.0

    async def _recall_user_preferences(self, user_id: str, topic: str) -> str:
        """Recall relevant user domain facts and constraints from LongTermMemory."""
        if not self.memory_service:
            return ""

        try:
            results = await self.memory_service.recall(
                user_id=user_id,
                query=topic,
                top_k=3,
            )
            if not results:
                return ""

            lines = ["Recalled user preferences and domain constraints:"]
            for r in results:
                # MemorySearchResult holds record with content
                content = getattr(r, "content", None)
                if not content and hasattr(r, "record"):
                    content = getattr(r.record, "content", str(r.record))
                if content:
                    lines.append(f"- {content}")
            return "\n".join(lines)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Failed to recall user preferences from memory: %s", exc)
            return ""

    async def _pin_decomposition_goal(
        self,
        user_id: str,
        session_id: str,
        topic: str,
        plan: ResearchDecomposition,
    ) -> None:
        """Store the active research decomposition in ShortTermMemory as a pinned goal."""
        if not self.memory_service:
            return

        try:
            goal_text = (
                f"Research Topic: {topic} | Effort: {plan.effort_level.value} | "
                f"Workers: {plan.worker_count} | Briefs: {', '.join(b.brief_id for b in plan.worker_briefs)}"
            )
            await self.memory_service.pin_goal(
                user_id=user_id,
                session_id=session_id,
                goal=goal_text,
                metadata={
                    "effort_level": plan.effort_level.value,
                    "worker_count": plan.worker_count,
                },
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Failed to pin decomposition goal to short-term memory: %s", exc)

    async def decompose(
        self,
        topic: str,
        user_id: str | None = None,
        session_id: str | None = None,
        effort_override: EffortLevel | None = None,
        shared_budget: SharedRunBudget | None = None,
    ) -> ResearchDecomposition:
        """Analyze a research topic, classify effort scaling, and generate typed worker briefs.

        Args:
            topic: The research question or topic to investigate.
            user_id: Optional user identifier for memory personalization.
            session_id: Optional session identifier for short-term working context.
            effort_override: Optional forced effort level (DIRECT or PARALLEL_DECOMPOSED).
            shared_budget: Optional shared run budget to record exact token consumption.

        Returns:
            Validated ResearchDecomposition containing typed worker briefs.

        Raises:
            DecompositionError: If decomposition generation or validation fails.
        """
        trimmed_topic = topic.strip()
        if not trimmed_topic:
            raise DecompositionError("Cannot decompose an empty research topic.")

        # 1. Recall user preferences from LongTermMemory if user_id is provided
        user_prefs = ""
        if user_id:
            user_prefs = await self._recall_user_preferences(user_id=user_id, topic=trimmed_topic)

        # 2. Render versioned prompt templates
        effort_mode_str = (
            effort_override.value if effort_override else self.config.default_effort_level
        )
        system_prompt, user_prompt = self.prompt_manager.render_decomposition_prompts(
            topic=trimmed_topic,
            max_workers=self.config.max_workers,
            effort_mode=effort_mode_str,
            user_preferences=user_prefs,
            per_worker_max_steps=self.config.per_worker_max_steps,
            per_worker_timeout_seconds=self.config.per_worker_timeout_seconds,
            version=self.config.prompt_version,
        )

        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        _logger.info(
            "Calling LLM for research decomposition on topic '%s' (effort_mode=%s, max_workers=%d)",
            trimmed_topic,
            effort_mode_str,
            self.config.max_workers,
        )

        # 3. Request structured completion via LLMService with transient retry
        max_attempts = 3
        llm_response = None
        for attempt in range(1, max_attempts + 1):
            try:
                llm_response = await self.llm_service.complete_structure(
                    prompt=full_prompt,
                    response_schema=ResearchDecomposition,
                    provider=self.config.default_provider,
                    model=self.config.default_model,
                )
            except Exception as exc:
                if attempt == max_attempts:
                    raise DecompositionError(
                        f"LLM decomposition completion call failed: {exc}"
                    ) from exc
                await asyncio.sleep(2.0 * attempt)
                continue

            err = getattr(llm_response, "error", None)
            if err is not None and isinstance(getattr(err, "message", None), str):
                err_msg = err.message.lower()
                is_transient = any(
                    k in err_msg
                    for k in [
                        "high demand",
                        "rate limit",
                        "503",
                        "429",
                        "resource exhausted",
                        "try again",
                    ]
                )
                if is_transient and attempt < max_attempts:
                    _logger.warning(
                        "Transient LLM error on decomposition attempt %d/%d: %s. Retrying in %.1fs...",
                        attempt,
                        max_attempts,
                        err.message,
                        3.0 * attempt,
                    )
                    await asyncio.sleep(3.0 * attempt)
                    continue
                raise DecompositionError(f"LLM decomposition error: {err.message}")
            break

        if llm_response is None:
            raise DecompositionError("LLM decomposition failed to produce a response.")

        plan: Any = getattr(llm_response, "structured_response", None)
        if not isinstance(plan, ResearchDecomposition):
            raise DecompositionError(
                f"LLM failed to return a valid ResearchDecomposition instance; got {type(plan).__name__}"
            )

        # Ensure worker_count aligns with length of briefs
        plan.worker_count = len(plan.worker_briefs)

        # 4. Extract real token usage and record exact expenditure
        usage = getattr(llm_response, "usage", {}) or {}
        in_tokens = (
            usage.get("input_tokens", 0)
            if isinstance(usage, dict)
            else getattr(usage, "input_tokens", 0)
        )
        out_tokens = (
            usage.get("output_tokens", 0)
            if isinstance(usage, dict)
            else getattr(usage, "output_tokens", 0)
        )
        total_tokens = in_tokens + out_tokens
        cost_usd = self.pricing_calculator.calculate_cost(
            self.config.default_model, in_tokens, out_tokens
        )
        self.last_input_tokens = in_tokens
        self.last_output_tokens = out_tokens
        self.last_total_tokens = total_tokens
        self.last_cost_usd = cost_usd

        plan.input_tokens = in_tokens
        plan.output_tokens = out_tokens

        if shared_budget is not None and total_tokens > 0:
            await shared_budget.record_usage(tokens=total_tokens, cost_usd=cost_usd)

        # 5. Validate decomposition against structural and boundary rules
        self.validator.validate(
            decomposition=plan,
            max_workers=self.config.max_workers,
            effort_override=effort_override,
        )

        # 6. Persist decomposition plan as pinned goal in ShortTermMemory
        if user_id and session_id:
            await self._pin_decomposition_goal(
                user_id=user_id,
                session_id=session_id,
                topic=trimmed_topic,
                plan=plan,
            )

        _logger.info(
            "Successfully decomposed research topic into %d briefs (%s effort)",
            plan.worker_count,
            plan.effort_level.value,
        )
        return plan
