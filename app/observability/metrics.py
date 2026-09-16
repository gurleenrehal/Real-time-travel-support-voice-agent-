"""
Per-stage latency instrumentation.

Uses time.perf_counter() (monotonic) to measure each pipeline stage
independently: STT, retrieval, tool execution, LLM generation, TTS, and
the total end-to-end turn latency. Metrics are kept in a small in-memory
ring buffer that backs the GET /metrics endpoint -- this is a portfolio
project, not a production metrics stack (no Prometheus/Grafana export is
implemented; see docs/ARCHITECTURE.md for what a production version would
add).
"""
from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class LatencyTimer:
    """Tracks named stage durations for a single turn, in milliseconds."""

    stages: dict[str, float] = field(default_factory=dict)
    _start_total: float = field(default_factory=time.perf_counter)

    @contextmanager
    def measure(self, stage_name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[stage_name] = (time.perf_counter() - start) * 1000.0

    def total_ms(self) -> float:
        return (time.perf_counter() - self._start_total) * 1000.0

    def as_dict(self) -> dict[str, float]:
        out = dict(self.stages)
        out["total_ms"] = self.total_ms()
        return out


class MetricsStore:
    """Small in-memory store of recent turn latencies, for GET /metrics."""

    def __init__(self, maxlen: int = 200) -> None:
        self._turns: deque[dict] = deque(maxlen=maxlen)

    def record_turn(self, latency: dict[str, float]) -> None:
        self._turns.append(latency)

    def summary(self) -> dict:
        if not self._turns:
            return {"turn_count": 0}
        keys = {k for turn in self._turns for k in turn}
        summary: dict[str, float] = {}
        for key in keys:
            values = [t[key] for t in self._turns if key in t]
            if values:
                summary[f"{key}_avg"] = sum(values) / len(values)
                summary[f"{key}_max"] = max(values)
        summary["turn_count"] = len(self._turns)
        return summary


metrics_store = MetricsStore()


# --------------------------------------------------------------------------
# Per-turn timer registry.
#
# LangGraph nodes are plain functions of `state`; they don't have direct
# access to the LatencyTimer created in the FastAPI handler. Rather than
# smuggle a non-serializable object through the typed state (which
# LangGraph may copy/merge), each turn's timer is registered here by
# turn_id and looked up from inside the nodes that need to measure a
# stage. This is what makes "retrieval_ms" / "tool_ms" / "llm_ms" in the
# final metrics genuinely measured inside the stage that does the work,
# rather than measured around the whole graph.invoke() call.
# --------------------------------------------------------------------------
_active_timers: dict[str, LatencyTimer] = {}


def register_timer(turn_id: str, timer: LatencyTimer) -> None:
    _active_timers[turn_id] = timer


def get_timer(turn_id: str) -> LatencyTimer | None:
    return _active_timers.get(turn_id)


def release_timer(turn_id: str) -> None:
    _active_timers.pop(turn_id, None)
