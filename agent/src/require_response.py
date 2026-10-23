"""Exactly-once client state for MiniCPM response-required turns.

Does not change PCM timing. Extra websocket fields are tiny JSON flags.
When disabled, every method is a no-op besides optional turn_id tracing.
"""

from __future__ import annotations

import os
import time
from typing import Any

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}
UNANSWERED_MS = int(os.environ.get("MINICPM_UNANSWERED_MS", "8000"))


def require_response_enabled(raw: str | None = None) -> bool:
    value = (raw if raw is not None else os.environ.get("MINICPM_REQUIRE_RESPONSE", "1")).strip().lower()
    if not value:
        return True
    if value in _FALSY:
        return False
    return value in _TRUTHY or True


class TurnResponseTracker:
    """Arm a VAD-completed turn until the first response, then never again."""

    def __init__(self, *, enabled: bool | None = None, unanswered_ms: int = UNANSWERED_MS) -> None:
        self.enabled = require_response_enabled() if enabled is None else bool(enabled)
        self.unanswered_ms = max(1, int(unanswered_ms))
        self.turn_id = 0
        self.state = "idle"  # idle | speech | waiting | answered | cancelled
        self.answered = False
        self.unanswered_emitted = False
        self.t_speech = 0.0
        self.t_terminal = 0.0
        self.t_commit = 0.0
        self.t_response = 0.0
        self.t_first_audio = 0.0
        self.response_id = ""
        self.cancel_reason = ""
        self.marked_appends = 0
        self.unmarked_appends = 0
        self.extra_bytes_total = 0
        self.last_extra_bytes = 0
        self.marker_logged = False
        self.listen_while_waiting = 0
        self.attach_us_max = 0
        self.last_attach_us = 0

    def reset_session(self, *, reason: str = "reset") -> dict[str, Any]:
        snapshot = self.snapshot(reason=reason)
        self.state = "idle"
        self.answered = False
        self.unanswered_emitted = False
        self.t_speech = 0.0
        self.t_terminal = 0.0
        self.t_commit = 0.0
        self.t_response = 0.0
        self.t_first_audio = 0.0
        self.response_id = ""
        self.cancel_reason = reason
        self.marked_appends = 0
        self.unmarked_appends = 0
        self.extra_bytes_total = 0
        self.last_extra_bytes = 0
        self.marker_logged = False
        self.listen_while_waiting = 0
        self.attach_us_max = 0
        self.last_attach_us = 0
        return snapshot

    def on_speech_start(self, now: float | None = None) -> dict[str, Any]:
        now = time.monotonic() if now is None else now
        previous = self.snapshot(reason="new_speech")
        self.turn_id += 1
        self.state = "speech"
        self.answered = False
        self.unanswered_emitted = False
        self.t_speech = now
        self.t_terminal = 0.0
        self.t_commit = 0.0
        self.t_response = 0.0
        self.t_first_audio = 0.0
        self.response_id = ""
        self.cancel_reason = ""
        self.marked_appends = 0
        self.unmarked_appends = 0
        self.extra_bytes_total = 0
        self.last_extra_bytes = 0
        self.marker_logged = False
        self.listen_while_waiting = 0
        self.attach_us_max = 0
        self.last_attach_us = 0
        return previous

    def mark_terminal(self, now: float | None = None) -> bool:
        """Idempotent: speech -> waiting. Returns True on the first call."""
        if self.state != "speech":
            return False
        self.state = "waiting"
        self.t_terminal = time.monotonic() if now is None else now
        return True

    def mark_commit(self, now: float | None = None) -> None:
        if self.state in {"speech", "waiting"} and self.t_commit <= 0.0:
            if self.state == "speech":
                self.mark_terminal(now)
            self.t_commit = time.monotonic() if now is None else now

    def waiting(self) -> bool:
        return self.enabled and self.state == "waiting" and not self.answered

    def from_commit_ms(self, now: float | None = None) -> int:
        if self.t_commit <= 0:
            return -1
        return _age_ms(self.t_commit, time.monotonic() if now is None else now)

    def marker_fields(self, *, is_speech: bool = False) -> dict[str, Any]:
        del is_speech
        if not self.enabled or self.turn_id <= 0:
            self.last_extra_bytes = 0
            return {}
        fields: dict[str, Any] = {"turn_id": self.turn_id}
        extra = 18
        if self.waiting():
            fields["response_required"] = True
            extra += 24
            self.marked_appends += 1
        else:
            self.unmarked_appends += 1
        self.last_extra_bytes = extra
        self.extra_bytes_total += extra
        return fields

    def note_attach_us(self, attach_us: int) -> None:
        value = int(attach_us)
        self.last_attach_us = value
        if value > self.attach_us_max:
            self.attach_us_max = value

    def note_listen_while_waiting(self) -> int:
        if not self.waiting():
            return self.listen_while_waiting
        self.listen_while_waiting += 1
        return self.listen_while_waiting

    def consume_first_marker(self) -> bool:
        if self.marker_logged or not self.waiting():
            return False
        self.marker_logged = True
        return True

    def hot_fields(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "req_state": self.state,
            "waiting": int(self.waiting()),
            "marked": int(bool(self.last_extra_bytes and self.waiting())),
        }

    def on_response_created(self, response_id: str = "", now: float | None = None) -> str:
        """Return '' on first response, 'duplicate' if another response already answered this turn."""
        now = time.monotonic() if now is None else now
        rid = str(response_id or "")
        if self.answered or self.state == "answered":
            if rid and self.response_id and rid != self.response_id:
                return "duplicate"
            if rid and not self.response_id:
                self.response_id = rid
            return ""
        self.answered = True
        self.state = "answered"
        self.t_response = now
        self.response_id = rid
        return ""

    def on_first_audio(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if self.t_first_audio <= 0.0:
            self.t_first_audio = now
        if not self.answered:
            self.on_response_created(self.response_id, now=now)

    def on_cancel(self, reason: str, now: float | None = None) -> dict[str, Any]:
        del now
        if self.state in {"idle", "answered", "cancelled"}:
            self.cancel_reason = reason
            return self.snapshot(reason=reason)
        self.state = "cancelled"
        self.cancel_reason = reason
        return self.snapshot(reason=reason)

    def maybe_unanswered(self, now: float | None = None) -> dict[str, Any] | None:
        if not self.waiting() or self.unanswered_emitted:
            return None
        now = time.monotonic() if now is None else now
        start = self.t_commit or self.t_terminal or self.t_speech
        if start <= 0.0:
            return None
        age_ms = int((now - start) * 1000)
        if age_ms < self.unanswered_ms:
            return None
        self.unanswered_emitted = True
        snap = self.snapshot(reason="deadline")
        snap["unanswered_ms"] = age_ms
        return snap

    def snapshot(self, **extra: Any) -> dict[str, Any]:
        now = time.monotonic()
        out: dict[str, Any] = {
            "turn_id": self.turn_id,
            "state": self.state,
            "enabled": int(self.enabled),
            "answered": int(self.answered),
            "waiting": int(self.waiting()),
            "response_id": self.response_id[:16],
            "from_speech_ms": _age_ms(self.t_speech, now),
            "from_terminal_ms": _age_ms(self.t_terminal, now),
            "from_commit_ms": _age_ms(self.t_commit, now),
            "from_response_ms": _age_ms(self.t_response, now),
            "cancel_reason": self.cancel_reason,
            "marked_appends": self.marked_appends,
            "unmarked_appends": self.unmarked_appends,
            "extra_bytes_total": self.extra_bytes_total,
            "listen_while_waiting": self.listen_while_waiting,
            "attach_us_max": self.attach_us_max,
        }
        out.update(extra)
        return out


def _age_ms(stamp: float, now: float) -> int:
    if stamp <= 0:
        return -1
    return int((now - stamp) * 1000)
