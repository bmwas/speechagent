"""Client-only duplex turn liveness. No protocol or model-server changes."""

from __future__ import annotations

import os
from typing import Any

DECISION_TIMEOUT_MS = int(os.environ.get("DUPLEX_DECISION_TIMEOUT_MS", "8000"))


def _age_ms(stamp: float, now: float) -> int:
    if stamp <= 0:
        return -1
    return int((now - stamp) * 1000)


class TurnWatchdog:
    """Arm on commit; clear on any terminal model decision; timeout if none."""

    def __init__(self, timeout_ms: int = DECISION_TIMEOUT_MS) -> None:
        self.timeout_ms = max(1000, int(timeout_ms))
        self.generation = 0
        self.armed = False
        self.t_commit = 0.0
        self.t_decision = 0.0
        self.decision = ""
        self.last_timeout_reason = ""

    def note_session(self) -> int:
        """Count a new duplex session after a successful handshake."""
        self.generation += 1
        self.armed = False
        self.t_commit = 0.0
        self.t_decision = 0.0
        self.decision = ""
        self.last_timeout_reason = ""
        return self.generation

    def reset_turn(self) -> None:
        self.armed = False
        self.t_commit = 0.0
        self.t_decision = 0.0
        self.decision = ""

    def arm(self, now: float) -> None:
        self.armed = True
        self.t_commit = now
        self.t_decision = 0.0
        self.decision = ""
        self.last_timeout_reason = ""

    def clear(self, now: float, decision: str) -> bool:
        """Clear an armed turn. Returns True if this was the waiting decision."""
        was_armed = self.armed
        self.armed = False
        self.t_decision = now
        self.decision = decision[:40]
        return was_armed

    def poll(self, now: float) -> str:
        if not self.armed or self.t_commit <= 0:
            return ""
        wait_ms = _age_ms(self.t_commit, now)
        if wait_ms < self.timeout_ms:
            return ""
        self.armed = False
        self.last_timeout_reason = f"decision_timeout_{wait_ms}ms"
        return self.last_timeout_reason

    def snapshot(self, now: float) -> dict[str, Any]:
        return {
            "turn_waiting": self.armed,
            "commit_age_ms": _age_ms(self.t_commit, now),
            "decision_age_ms": _age_ms(self.t_decision, now),
            "session_generation": self.generation,
            "last_decision": self.decision,
            "decision_timeout_ms": self.timeout_ms,
        }
