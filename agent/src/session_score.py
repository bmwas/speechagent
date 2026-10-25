"""Per-turn latency scoreboard with p50/p95 for the /patient session lab."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# Cap turn samples to bound memory on long calls.
MAX_TURNS = 200


@dataclass
class TurnSample:
    """One completed turn's latency figures."""

    ttfa_ms: float = -1.0  # Time to first audio (anchor-dependent)
    ttfa_anchor: str = ""  # "query" (Qwen), "speak" (vLLM), "speak" (native)
    ttft_ms: float = -1.0  # Time to first text
    rtf: float = -1.0  # Real-time factor (audio ms / wall ms)
    gap_ms: float = -1.0  # Longest inter-delta silence
    audio_ms: float = 0.0  # Total audio produced this turn
    prebuffer_ms: float = -1.0  # Jitter buffer wait before first frame
    e2e_ms: float = -1.0  # TTFA + prebuffer (user-perceived latency)


@dataclass
class SessionScore:
    """Accumulates turn samples and computes session-level stats."""

    turns: list[TurnSample] = field(default_factory=list)
    _t_start: float = field(default_factory=time.monotonic)
    # Session integrals (updated by the 1 Hz health loop)
    user_speech_s: float = 0.0
    agent_audible_s: float = 0.0
    stalls: int = 0
    reconnects: int = 0
    barges: int = 0
    mic_gaps: int = 0

    def add_turn(self, sample: TurnSample) -> None:
        """Append a completed turn. Drops oldest if over cap."""
        self.turns.append(sample)
        if len(self.turns) > MAX_TURNS:
            self.turns = self.turns[-MAX_TURNS:]

    def reset(self) -> None:
        """Clear all state for a new session."""
        self.turns.clear()
        self._t_start = time.monotonic()
        self.user_speech_s = 0.0
        self.agent_audible_s = 0.0
        self.stalls = 0
        self.reconnects = 0
        self.barges = 0
        self.mic_gaps = 0

    @property
    def duration_s(self) -> float:
        return time.monotonic() - self._t_start

    @property
    def talk_ratio(self) -> float:
        """Agent audible time / session time."""
        d = self.duration_s
        return round(self.agent_audible_s / d, 3) if d > 0 else 0.0

    def _percentile(self, values: list[float], p: float) -> float:
        """Return the p-th percentile (0–100) of a sorted list."""
        if not values:
            return -1.0
        n = len(values)
        if n == 1:
            return values[0]
        k = (n - 1) * p / 100.0
        f = int(k)
        c = f + 1 if f + 1 < n else f
        return values[f] + (values[c] - values[f]) * (k - f)

    def _stats_for(self, attr: str) -> dict[str, float]:
        """Return mean, p50, p95 for an attribute across all turns."""
        vals = [getattr(t, attr) for t in self.turns if getattr(t, attr, -1) >= 0]
        if not vals:
            return {"mean": -1.0, "p50": -1.0, "p95": -1.0}
        vals_sorted = sorted(vals)
        mean = sum(vals) / len(vals)
        return {
            "mean": round(mean, 1),
            "p50": round(self._percentile(vals_sorted, 50), 1),
            "p95": round(self._percentile(vals_sorted, 95), 1),
        }

    def snapshot(self) -> dict[str, Any]:
        """Return full session stats for /api/live."""
        last = self.turns[-1] if self.turns else None
        return {
            "n": len(self.turns),
            "duration_s": round(self.duration_s, 1),
            "user_speech_s": round(self.user_speech_s, 1),
            "agent_audible_s": round(self.agent_audible_s, 1),
            "talk_ratio": self.talk_ratio,
            "stalls": self.stalls,
            "reconnects": self.reconnects,
            "barges": self.barges,
            "mic_gaps": self.mic_gaps,
            "ttfa": self._stats_for("ttfa_ms"),
            "ttft": self._stats_for("ttft_ms"),
            "rtf": self._stats_for("rtf"),
            "gap": self._stats_for("gap_ms"),
            "prebuffer": self._stats_for("prebuffer_ms"),
            "e2e": self._stats_for("e2e_ms"),
            "last_turn": {
                "ttfa_ms": last.ttfa_ms if last else -1,
                "ttfa_anchor": last.ttfa_anchor if last else "",
                "ttft_ms": last.ttft_ms if last else -1,
                "rtf": last.rtf if last else -1,
                "gap_ms": last.gap_ms if last else -1,
                "audio_ms": last.audio_ms if last else 0,
                "prebuffer_ms": last.prebuffer_ms if last else -1,
                "e2e_ms": last.e2e_ms if last else -1,
            } if last else None,
        }
