"""Structured one-line JSON traces for Loki (msestream agent_trace pattern).

Every call to :func:`trace` emits exactly one stdout line::

    SA_TRACE {"ts":..., "agent":"speech", "event":"session.open", ...}

Promtail (obs/promtail.yml) matches the ``SA_TRACE `` prefix, parses the JSON
tail, and promotes ``agent`` / ``event`` to Loki labels. ``session_id`` and
``trace_id`` go to structured metadata.

Never raises. PCM/JPEG bytes must not be passed as fields.
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
import uuid
from contextvars import ContextVar
from typing import Any

_LOG = logging.getLogger("sa.trace")
_HOSTNAME = os.environ.get("HOSTNAME", "")
_trace_id_var: ContextVar[str] = ContextVar("sa_trace_id", default="")
_session_id_var: ContextVar[str] = ContextVar("sa_session_id", default="")
_seq: dict[str, int] = {}

_TRUTHY = {"1", "true", "yes", "on"}
TRACE_PREFIX = "SA_TRACE "
_DISABLED = (os.environ.get("SA_TRACE_DISABLED") or "").strip().lower() in _TRUTHY
_DEBUG = (os.environ.get("TRACE_DEBUG") or "1").strip().lower() in _TRUTHY


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def set_trace_id(tid: str = "") -> str:
    tid = tid or new_trace_id()
    _trace_id_var.set(tid)
    return tid


def get_trace_id() -> str:
    return _trace_id_var.get()


def set_session_id(sid: str) -> str:
    _session_id_var.set(sid or "")
    return sid


def get_session_id() -> str:
    return _session_id_var.get()


def debug_enabled() -> bool:
    return _DEBUG


def _next_seq(agent: str) -> int:
    n = _seq.get(agent, 0) + 1
    _seq[agent] = n
    return n


def trace_dbg(agent: str, event: str, **fields: Any) -> None:
    if not _DEBUG:
        return
    trace(agent, event, level="debug", **fields)


def trace(agent: str, event: str, **fields: Any) -> None:
    if _DISABLED:
        return
    payload: dict[str, Any] = {
        "ts": time.time(),
        "agent": agent,
        "event": event,
    }
    tid = _trace_id_var.get()
    sid = _session_id_var.get()
    if tid:
        payload["trace_id"] = tid
    if sid:
        payload["session_id"] = sid
    if _HOSTNAME:
        payload["host"] = _HOSTNAME
    if fields:
        payload.update(fields)
    payload["seq"] = _next_seq(agent)
    try:
        line = TRACE_PREFIX + json.dumps(payload, default=str, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        line = TRACE_PREFIX + json.dumps(
            {
                "ts": time.time(),
                "agent": agent,
                "event": "trace.serialize_error",
                "trace_id": tid,
                "error": str(exc)[:160],
            },
            separators=(",", ":"),
        )
    _LOG.info(line)


def trace_exc(agent: str, event: str, exc: BaseException | None = None, **fields: Any) -> None:
    if exc is None:
        import sys as _sys

        exc = _sys.exc_info()[1]
    err_type = type(exc).__name__ if exc is not None else ""
    err_msg = str(exc)[:300] if exc is not None else ""
    try:
        tb = traceback.format_exc()
    except Exception:
        tb = ""
    if tb and tb.strip() == "NoneType: None":
        tb = ""
    trace(
        agent,
        event,
        level="error",
        error_type=err_type,
        error=err_msg,
        traceback=tb[-1500:] if tb else "",
        **fields,
    )


class trace_turn:
    """Scopes a trace_id and emits ``<event>.start`` / ``<event>.end``."""

    def __init__(self, agent: str, event: str = "turn", trace_id: str = "", **fields: Any):
        self.agent = agent
        self.event = event
        self.trace_id = trace_id
        self.fields = fields
        self._token = None
        self._t0 = 0.0

    def __enter__(self) -> str:
        tid = self.trace_id or new_trace_id()
        self._token = _trace_id_var.set(tid)
        self._t0 = time.monotonic()
        trace(self.agent, f"{self.event}.start", **self.fields)
        return tid

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            lat_ms = int((time.monotonic() - self._t0) * 1000)
            if exc_type is not None:
                trace(
                    self.agent,
                    f"{self.event}.error",
                    latency_ms=lat_ms,
                    error_type=exc_type.__name__,
                    error=str(exc)[:200] if exc else "",
                )
            else:
                trace(self.agent, f"{self.event}.end", latency_ms=lat_ms)
        finally:
            if self._token is not None:
                _trace_id_var.reset(self._token)
