"""MiniCPM response-required policy. Pure functions; no vLLM import.

Copied into the vLLM-Omni package at model-server startup and used by the
version-pinned Stage-0 sampling patch. Tests import this file directly.

Logging is gated so Stage-0 sampling never writes a line per audio unit.
Immediate logs are once per (turn, reason) for speak/override/duplicate/
marker receipt. Everything else rolls into a 1 Hz summary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}
LOG_PREFIX = "SA_REQUIRE_RESPONSE"
IMMEDIATE_REASONS = frozenset(
    {
        "listen_overridden",
        "natural_speak",
        "duplicate",
        "marker_receipt",
    }
)
_ENABLED_CACHE: bool | None = None


def require_response_enabled(raw: str | None = None) -> bool:
    if raw is None:
        return True
    value = str(raw).strip().lower()
    if not value:
        return True
    if value in _FALSY:
        return False
    return True


def env_require_response_enabled() -> bool:
    global _ENABLED_CACHE
    if _ENABLED_CACHE is None:
        import os

        _ENABLED_CACHE = require_response_enabled(os.environ.get("MINICPM_REQUIRE_RESPONSE"))
    return _ENABLED_CACHE


def reset_enabled_cache() -> None:
    global _ENABLED_CACHE
    _ENABLED_CACHE = None


@dataclass
class RequireLogDecision:
    immediate: dict[str, Any] | None
    summary: dict[str, Any] | None


class RequireLogGate:
    """Keep GPU sampling off the logging path.

    ``offer`` never formats strings and never I/O. Callers may log the
    returned payloads on a slow path. Immediate events fire once per
    ``(turn_id, reason)``. Counts flush at most once per ``period_s``.
    """

    def __init__(self, *, period_s: float = 1.0) -> None:
        self.period_s = max(0.2, float(period_s))
        self._counts: dict[str, int] = {}
        self._logged: set[tuple[Any, str]] = set()
        self._t0 = time.monotonic()
        self._n = 0

    def offer(self, reason: str, turn_id: Any = None, **fields: Any) -> RequireLogDecision:
        self._n += 1
        self._counts[reason] = self._counts.get(reason, 0) + 1
        immediate = None
        key = (turn_id, reason)
        if reason in IMMEDIATE_REASONS and key not in self._logged:
            self._logged.add(key)
            if len(self._logged) > 4096:
                self._logged.clear()
            immediate = {"kind": "event", "reason": reason, "turn_id": turn_id, **fields}
        summary = None
        now = time.monotonic()
        if (now - self._t0) >= self.period_s:
            summary = {
                "kind": "summary",
                "n": self._n,
                "window_s": round(now - self._t0, 3),
                "counts": dict(self._counts),
            }
            self._counts.clear()
            self._n = 0
            self._t0 = now
        return RequireLogDecision(immediate=immediate, summary=summary)


def payload_turn_id(payload: dict[str, Any] | None) -> int | str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("turn_id")
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError:
            return value.strip()
    return None


def payload_response_required(payload: dict[str, Any] | None) -> bool:
    return isinstance(payload, dict) and payload.get("response_required") is True


def copy_require_fields(src: dict[str, Any], dest: dict[str, Any]) -> None:
    if src.get("response_required") is True:
        dest["response_required"] = True
    turn_id = payload_turn_id(src)
    if turn_id is not None:
        dest["turn_id"] = turn_id


def latch_from_payload(
    payload: dict[str, Any] | None,
    *,
    currently_latched: int | str | None,
) -> int | str | None:
    if not payload_response_required(payload):
        return currently_latched
    turn_id = payload_turn_id(payload)
    return currently_latched if turn_id is None else turn_id


@dataclass(frozen=True)
class RequireDecision:
    override_listen: bool
    block_force_listen: bool
    reason: str
    turn_id: int | str | None
    required: bool


def decide_require_response(
    *,
    enabled: bool,
    sampled_is_listen: bool | None,
    force_listen: bool,
    is_speech: bool,
    response_required: bool,
    pending_speech_context: bool,
    current_turn_ended: bool,
    pending_speech_response_open: bool,
    turn_id: int | str | None,
    opened_turn_ids: set[Any],
    latched_turn_id: int | str | None,
) -> RequireDecision:
    """Decide whether this Stage-0 sample must speak.

    Native duplex listening is preserved while the user is still speaking
    (unmarked speech payloads) and while a response is already open.
    """
    active_turn = turn_id if turn_id is not None else latched_turn_id
    required = bool(
        response_required or (latched_turn_id is not None and (turn_id is None or turn_id == latched_turn_id))
    )
    if not enabled:
        return RequireDecision(False, False, "disabled", active_turn, required)

    if is_speech and not response_required:
        return RequireDecision(False, False, "user_speaking", active_turn, required)

    response_open = pending_speech_response_open or not current_turn_ended
    if response_open:
        return RequireDecision(False, False, "response_active", active_turn, required)

    if not required:
        return RequireDecision(False, False, "not_required", active_turn, required)

    if not pending_speech_context:
        return RequireDecision(False, False, "no_pending_speech", active_turn, required)

    if active_turn is not None and active_turn in opened_turn_ids:
        return RequireDecision(False, False, "duplicate", active_turn, required)

    if force_listen and not required:
        return RequireDecision(False, False, "force_listen", active_turn, required)

    if sampled_is_listen is None:
        return RequireDecision(False, True, "armed", active_turn, required)
    if sampled_is_listen is False:
        return RequireDecision(False, True, "natural_speak", active_turn, required)
    return RequireDecision(True, True, "listen_overridden", active_turn, required)
