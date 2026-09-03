from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    """Represents one timed operation inside a request."""

    name: str

    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    start_time: float = field(default_factory=time.perf_counter)

    end_time: float | None = None

    status: str = "running"

    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def latency_ms(self) -> float | None:
        if self.end_time is None:
            return None

        return (self.end_time - self.start_time) * 1000

    def set(self, **attributes: Any) -> None:
        self.attributes.update(attributes)

    def finish(self, status: str = "success") -> float:
        if self.end_time is None:
            self.end_time = time.perf_counter()

        self.status = status

        return self.latency_ms or 0.0

    def fail(self, error_type: str | None = None) -> float:
        if error_type is not None:
            self.set(error_type=error_type)

        return self.finish(status="error")

    def __enter__(self) -> Span:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        if exc_type is not None:
            self.fail(error_type=exc_type.__name__)
        else:
            self.finish()


class Tracer:
    """Creates timed spans."""

    def start_span(self, name: str) -> Span:
        return Span(name=name)


tracer = Tracer()
