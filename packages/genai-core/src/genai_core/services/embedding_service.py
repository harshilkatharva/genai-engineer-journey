from __future__ import annotations

import contextlib
import time

from ..core import get_settings
from ..embedding import (
    EmbeddingProvider,
    GoogleEmbeddingProvider,
    LocalEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from ..exceptions import (
    EmbeddingDimensionError,
    EmbeddingError,
    EmbeddingProviderError,
    EmbeddingTimeoutError,
)
from ..models import EmbeddingRequest, EmbeddingResponse
from ..observability import ObservabilityEvent, get_logger, safe_log
from ..resilience import (
    RateLimiter,
    RetryPolicy,
    estimate_tokens,
    execute_with_retry,
    extract_retry_after,
)

_logger = get_logger("genai_core.services.embedding_service")


class EmbeddingService:
    """Provider-independent entry point for text vectorization, batch embedding,

    and resilient rate-limited execution.
    """

    def __init__(
        self,
        providers: dict[str, EmbeddingProvider] | None = None,
        provider: EmbeddingProvider | str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        batch_size: int | None = None,
        rate_limiter: RateLimiter | None = None,
        retry_policy: RetryPolicy | None = None,
        rpm: int | None = None,
        tpm: int | None = None,
        max_retries: int | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        settings = get_settings()
        default_dim = getattr(settings, "default_embedding_dimensions", 384)
        self.dimensions: int = int(dimensions if dimensions is not None else default_dim)
        default_batch = getattr(settings, "default_embedding_batch_size", 32)
        self.batch_size: int = int(batch_size if batch_size is not None else default_batch)

        if provider is not None and not isinstance(provider, str):
            # Direct provider instance passed
            self.active_provider: EmbeddingProvider = provider
            self.provider_name: str = getattr(
                provider, "provider_name", provider.__class__.__name__
            )
            self.model: str = str(
                model
                or getattr(provider, "default_model", None)
                or getattr(settings, "default_embedding_model", "all-MiniLM-L6-v2")
            )
            self.providers: dict[str, EmbeddingProvider] = (
                providers if providers is not None else {self.provider_name: self.active_provider}
            )
        else:
            self.provider_name = str(
                provider or getattr(settings, "default_embedding_provider", "local")
            )
            self.model = str(
                model or getattr(settings, "default_embedding_model", "all-MiniLM-L6-v2")
            )
            if providers is not None:
                self.providers = providers
            else:
                self.providers = self._init_default_providers()

            active = self.providers.get(self.provider_name)
            if active is None:
                active = self._resolve_fallback_provider(self.provider_name)
            self.active_provider = active

        # Strictly validate requested dimensions during service initialization
        self.active_provider.validate_dimensions(self.dimensions)

        # Initialize resilience components
        effective_rpm = rpm
        effective_tpm = tpm
        if self.provider_name in {"openai", "google"}:
            effective_rpm = rpm or getattr(settings, "embedding_rate_limit_rpm", None)
            effective_tpm = tpm or getattr(settings, "embedding_rate_limit_tpm", None)

        self.rate_limiter = rate_limiter or RateLimiter(
            rpm=effective_rpm,
            tpm=effective_tpm,
            provider_name=self.provider_name,
        )

        eff_max_retries = (
            max_retries
            if max_retries is not None
            else getattr(settings, "embedding_max_retries", 3)
        )
        eff_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else getattr(settings, "embedding_timeout_seconds", 30.0)
        )
        eff_init_backoff = getattr(settings, "embedding_initial_backoff", 0.5)
        eff_max_backoff = getattr(settings, "embedding_max_backoff", 10.0)

        self.retry_policy = retry_policy or RetryPolicy(
            max_retries=eff_max_retries,
            initial_backoff=eff_init_backoff,
            max_backoff=eff_max_backoff,
            timeout_seconds=eff_timeout,
        )

    def _resolve_fallback_provider(self, name: str) -> EmbeddingProvider:
        """Attempt to instantiate a named provider directly to surface actionable errors."""
        if name == "local":
            local_prov = LocalEmbeddingProvider(model=self.model)
            self.providers["local"] = local_prov
            return local_prov
        if name == "openai":
            openai_prov = OpenAIEmbeddingProvider(
                model=self.model if "text-embedding" in self.model else None,
                dimensions=self.dimensions,
            )
            self.providers["openai"] = openai_prov
            return openai_prov
        if name == "google":
            google_prov = GoogleEmbeddingProvider(
                model=self.model if "embedding" in self.model else None,
                dimensions=self.dimensions,
            )
            self.providers["google"] = google_prov
            return google_prov
        raise EmbeddingProviderError(
            name,
            f"Unsupported embedding provider '{name}'.",
        )

    @property
    def provider(self) -> EmbeddingProvider:
        """Return the active embedding provider."""
        return self.active_provider

    def _init_default_providers(self) -> dict[str, EmbeddingProvider]:
        providers: dict[str, EmbeddingProvider] = {}
        with contextlib.suppress(Exception):
            providers["local"] = LocalEmbeddingProvider(model=self.model)
        with contextlib.suppress(Exception):
            providers["openai"] = OpenAIEmbeddingProvider(dimensions=self.dimensions)
        with contextlib.suppress(Exception):
            providers["google"] = GoogleEmbeddingProvider(dimensions=self.dimensions)
        return providers

    async def embed(
        self,
        request: EmbeddingRequest | None = None,
        *,
        input: str | list[str] | None = None,
        batch_size: int | None = None,
        normalize_embeddings: bool = True,
    ) -> EmbeddingResponse:
        """Process an embedding request with internal batching, rate-limiting, and order preservation."""
        if request is None:
            if input is None:
                raise ValueError("Input cannot be empty.")
            req = EmbeddingRequest(
                input=input,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
            )
        else:
            req = request

        start_time = time.perf_counter()

        # Validate input collection
        if isinstance(req.input, str):
            clean_input = req.input.strip()
            if not clean_input:
                raise ValueError("Input text cannot be empty or whitespace only.")
            texts = [req.input]
        elif isinstance(req.input, list):
            if not req.input:
                raise ValueError("Input cannot be empty.")
            for idx, item in enumerate(req.input):
                if not isinstance(item, str) or not item.strip():
                    raise ValueError(
                        f"Input text at index {idx} cannot be empty or whitespace only."
                    )
            texts = list(req.input)
        else:
            raise TypeError("Input must be a string or a list of strings.")

        # Validate batch size
        effective_batch_size = req.batch_size if req.batch_size is not None else self.batch_size
        if effective_batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")

        total_inputs = len(texts)
        batches_count = (total_inputs + effective_batch_size - 1) // effective_batch_size

        safe_log(
            _logger,
            20,  # INFO
            "Embedding request initiated",
            event=ObservabilityEvent.EMBEDDING_REQUEST_START,
            provider=self.provider_name,
            model=self.model,
            dimensions=self.dimensions,
            total_inputs=total_inputs,
            batch_size=effective_batch_size,
            batches_count=batches_count,
        )

        request_deadline = time.monotonic() + self.retry_policy.timeout_seconds
        accumulated_vectors: list[list[float]] = []
        accumulated_usage: dict[str, int] | None = None

        try:
            # Internal batching loop: slices inputs, preserving exact 1-to-1 order
            for start_idx in range(0, total_inputs, effective_batch_size):
                end_idx = min(start_idx + effective_batch_size, total_inputs)
                batch = texts[start_idx:end_idx]

                # Rate limiting pacing
                est_tokens = estimate_tokens(batch)
                waited = await self.rate_limiter.acquire(requests=1, tokens=est_tokens)
                if waited > 0:
                    safe_log(
                        _logger,
                        20,  # INFO
                        f"Embedding rate limit throttled request: paused for {waited:.2f}s",
                        event=ObservabilityEvent.EMBEDDING_RATE_LIMITED,
                        provider=self.provider_name,
                        wait_seconds=round(waited, 3),
                    )

                def _create_call(sub_batch: list[str]):
                    async def _call() -> list[list[float]]:
                        return await self.active_provider.embed_batch(
                            sub_batch,
                            normalize=req.normalize_embeddings,
                        )

                    return _call

                def _on_retry(attempt: int, exc: Exception, delay: float) -> None:
                    suggested_wait = extract_retry_after(exc)
                    if suggested_wait:
                        self.rate_limiter.record_rate_limit(suggested_wait)

                    safe_log(
                        _logger,
                        30,  # WARNING
                        f"Embedding batch retry {attempt}/{self.retry_policy.max_retries}: {exc}",
                        event=ObservabilityEvent.EMBEDDING_RETRY,
                        provider=self.provider_name,
                        attempt=attempt,
                        delay=round(delay, 3),
                        error=str(exc),
                    )

                batch_vectors = await execute_with_retry(
                    _create_call(batch),
                    self.retry_policy,
                    deadline=request_deadline,
                    on_retry=_on_retry,
                )

                if len(batch_vectors) != len(batch):
                    raise RuntimeError(
                        f"Provider '{self.provider_name}' returned {len(batch_vectors)} vectors "
                        f"for batch of size {len(batch)}."
                    )

                # Validate vector dimensions
                if batch_vectors and len(batch_vectors[0]) != self.dimensions:
                    raise EmbeddingDimensionError(
                        model=self.model,
                        requested=self.dimensions,
                        supported=len(batch_vectors[0]),
                    )

                accumulated_vectors.extend(batch_vectors)

                # Collect usage metadata if reported by provider (e.g. OpenAI)
                provider_usage = getattr(self.active_provider, "last_usage", None)
                if provider_usage and isinstance(provider_usage, dict):
                    if accumulated_usage is None:
                        accumulated_usage = {"prompt_tokens": 0, "total_tokens": 0}
                    accumulated_usage["prompt_tokens"] += int(
                        provider_usage.get("prompt_tokens", 0)
                    )
                    accumulated_usage["total_tokens"] += int(provider_usage.get("total_tokens", 0))

            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

            safe_log(
                _logger,
                20,  # INFO
                "Embedding request completed successfully",
                event=ObservabilityEvent.EMBEDDING_REQUEST_SUCCESS,
                provider=self.provider_name,
                model=self.model,
                dimensions=self.dimensions,
                total_inputs=total_inputs,
                latency_ms=latency_ms,
            )

            return EmbeddingResponse(
                embeddings=accumulated_vectors,
                model=self.model,
                provider=self.provider_name,
                dimension=self.dimensions,
                usage=accumulated_usage,  # None for local models, actual counts for hosted
                latency_ms=latency_ms,
            )

        except EmbeddingTimeoutError as err:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            safe_log(
                _logger,
                40,  # ERROR
                f"Embedding request timed out: {err}",
                event=ObservabilityEvent.EMBEDDING_TIMEOUT,
                provider=self.provider_name,
                model=self.model,
                dimensions=self.dimensions,
                latency_ms=latency_ms,
            )
            raise
        except (ValueError, TypeError, EmbeddingError):
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            safe_log(
                _logger,
                40,  # ERROR
                "Embedding request failed",
                event=ObservabilityEvent.EMBEDDING_REQUEST_FAILED,
                provider=self.provider_name,
                model=self.model,
                dimensions=self.dimensions,
                latency_ms=latency_ms,
            )
            raise
        except Exception as err:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            safe_log(
                _logger,
                40,  # ERROR
                f"Embedding request failed: {err}",
                exc_info=err,
                event=ObservabilityEvent.EMBEDDING_REQUEST_FAILED,
                provider=self.provider_name,
                model=self.model,
                dimensions=self.dimensions,
                latency_ms=latency_ms,
            )
            raise EmbeddingProviderError(self.provider_name, str(err)) from err

    async def embed_query(self, query: str) -> list[float]:
        """Convenience method returning a single float vector for a search query."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query cannot be empty.")

        response = await self.embed(EmbeddingRequest(input=clean_query, normalize_embeddings=True))
        return response.embeddings[0]

    async def embed_documents(
        self,
        documents: list[str],
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """Convenience method returning a list of float vectors for chunks or documents."""
        response = await self.embed(
            EmbeddingRequest(
                input=documents,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
        )
        return response.embeddings
