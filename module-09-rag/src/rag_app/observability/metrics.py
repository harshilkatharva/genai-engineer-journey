from __future__ import annotations

from collections import defaultdict


class Metrics:
    """Simple in-process metrics collector."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = defaultdict(int)

    def increment(
        self,
        name: str,
        value: int = 1,
    ) -> None:
        self._counters[name] += value

    def get(self, name: str) -> int:
        return self._counters[name]

    def snapshot(self) -> dict[str, int]:
        return dict(self._counters)


metrics = Metrics()
