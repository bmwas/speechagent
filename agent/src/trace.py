"""Structured one-line JSON traces for Loki (msestream agent_trace pattern).

Every call to :func:`trace` emits exactly one stdout line::

    SA_TRACE {"ts":..., "agent":"speech", "event":"session.open", ...}

Promtail (obs/promtail.yml) matches the ``SA_TRACE `` prefix, parses the JSON
tail, and promotes ``agent`` / ``event`` / ``lane`` to Loki labels.
``session_id`` and ``trace_id`` go to structured metadata.

Streaming / duplex events go through :func:`trace_stream` (``lane=stream``)
so Grafana can isolate the realtime pipeline without drowning in Cudagraph
INFO. High-frequency PCM/JPEG is aggregated by :class:`StreamMeter` (1 Hz).

Never raises. PCM/JPEG bytes must not be passed as fields.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import sys
import threading
import time
import traceback
import uuid
import atexit
from contextvars import ContextVar
from typing import Any

_LOG = logging.getLogger("sa.trace")
_HOSTNAME = os.environ.get("HOSTNAME", "")
_trace_id_var: ContextVar[str] = ContextVar("sa_trace_id", default="")
_session_id_var: ContextVar[str] = ContextVar("sa_session_id", default="")
_seq: dict[str, int] = {}

_TRUTHY = {"1", "true", "yes", "on"}
TRACE_PREFIX = "SA_TRACE "
STREAM_LANE = "stream"
_DISABLED = (os.environ.get("SA_TRACE_DISABLED") or "").strip().lower() in _TRUTHY
_DEBUG = (os.environ.get("TRACE_DEBUG") or "1").strip().lower() in _TRUTHY
_STREAM = (os.environ.get("STREAM_TRACE") or "1").strip().lower() in _TRUTHY
_HIRES = (os.environ.get("TRACE_HIRES") or "").strip().lower() in _TRUTHY
_ASYNC_WRITE = (os.environ.get("TRACE_ASYNC_WRITE") or "1").strip().lower() in _TRUTHY
_MONO_T0 = time.monotonic()
HIRES_PERIOD_S = 0.1
_TRACE_QUEUE_MAX = int(os.environ.get("TRACE_QUEUE_MAX", "20000"))
_TRACE_BATCH_MAX = int(os.environ.get("TRACE_BATCH_MAX", "256"))
_TRACE_FLUSH_MS = float(os.environ.get("TRACE_FLUSH_MS", "20"))
_trace_queue: queue.Queue[str | None] = queue.Queue(maxsize=max(_TRACE_QUEUE_MAX, 100))
_trace_writer: threading.Thread | None = None
_trace_writer_lock = threading.Lock()
_trace_dropped = 0


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


STREAM_PERIOD_S = HIRES_PERIOD_S if _HIRES else _env_float("STREAM_TRACE_PERIOD_S", 1.0)
# Fields that should be max'd (not summed) inside a StreamMeter window.
_METER_MAX_KEYS = {
    "rms",
    "qsize",
    "ring_ms",
    "mic_age_ms",
    "delta_age_ms",
    "append_age_ms",
    "open_thresh",
    "latency_ms",
    "gap_ms",
    "block_ms",
    "utt_ms",
    "stall_ms",
}
_METER_LAST_KEYS = {"rate", "src_rate", "sample_rate"}


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


def stream_enabled() -> bool:
    return _STREAM and not _DISABLED


def hires_enabled() -> bool:
    return _HIRES and not _DISABLED


def hires_period(normal_s: float) -> float:
    """Return 0.1 s when TRACE_HIRES is on, otherwise *normal_s*."""
    return HIRES_PERIOD_S if _HIRES else normal_s


def mono_ms() -> int:
    return int((time.monotonic() - _MONO_T0) * 1000)


def _trace_writer_loop() -> None:
    while True:
        try:
            first = _trace_queue.get(timeout=0.25)
        except queue.Empty:
            continue
        if first is None:
            _trace_queue.task_done()
            break
        batch = [first]
        if _TRACE_FLUSH_MS > 0:
            time.sleep(_TRACE_FLUSH_MS / 1000.0)
        while len(batch) < _TRACE_BATCH_MAX:
            try:
                line = _trace_queue.get_nowait()
            except queue.Empty:
                break
            if line is None:
                _trace_queue.task_done()
                break
            batch.append(line)
        try:
            sys.stdout.write("\n".join(batch) + "\n")
            sys.stdout.flush()
        except Exception:
            for line in batch:
                _LOG.info(line)
        finally:
            for _ in batch:
                _trace_queue.task_done()


def _ensure_trace_writer() -> None:
    global _trace_writer
    if _trace_writer is not None and _trace_writer.is_alive():
        return
    with _trace_writer_lock:
        if _trace_writer is not None and _trace_writer.is_alive():
            return
        _trace_writer = threading.Thread(
            target=_trace_writer_loop,
            name="sa-trace-writer",
            daemon=True,
        )
        _trace_writer.start()


def flush_traces(timeout_s: float = 1.0) -> None:
    """Best-effort drain used at shutdown and by diagnostics."""
    if not _ASYNC_WRITE or _trace_writer is None:
        return
    deadline = time.monotonic() + max(timeout_s, 0.0)
    while _trace_queue.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.005)


def _write_trace_line(line: str) -> bool:
    if not _ASYNC_WRITE:
        try:
            print(line, flush=True)
        except Exception:
            _LOG.info(line)
        return True
    _ensure_trace_writer()
    try:
        _trace_queue.put_nowait(line)
        return True
    except queue.Full:
        return False


atexit.register(flush_traces)


def _next_seq(agent: str) -> int:
    n = _seq.get(agent, 0) + 1
    _seq[agent] = n
    return n


def _clip(value: Any, n: int = 240) -> Any:
    if isinstance(value, str) and len(value) > n:
        return value[:n]
    return value


def trace_dbg(agent: str, event: str, **fields: Any) -> None:
    if not _DEBUG:
        return
    trace(agent, event, level="debug", **fields)


def trace_stream(agent: str, event: str, **fields: Any) -> None:
    """Realtime/duplex pipeline event. Promtail labels these ``lane=stream``."""
    if not _STREAM:
        return
    fields.setdefault("lane", STREAM_LANE)
    trace(agent, event, **fields)


def trace(agent: str, event: str, **fields: Any) -> None:
    global _trace_dropped
    if _DISABLED:
        return
    payload: dict[str, Any] = {
        "ts": time.time(),
        "mono_ms": mono_ms(),
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
    if _HIRES and _ASYNC_WRITE:
        payload["trace_queue_depth"] = _trace_queue.qsize()
    if _trace_dropped:
        payload["trace_queue_dropped"] = _trace_dropped
    if fields:
        for key, value in fields.items():
            payload[key] = _clip(value)
    payload["seq"] = _next_seq(agent)
    try:
        line = TRACE_PREFIX + json.dumps(payload, default=str, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        line = TRACE_PREFIX + json.dumps(
            {
                "ts": time.time(),
                "mono_ms": mono_ms(),
                "agent": agent,
                "event": "trace.serialize_error",
                "trace_id": tid,
                "error": str(exc)[:160],
            },
            separators=(",", ":"),
        )
    # Queue stdout writes off the realtime event loop. The writer preserves
    # ordering and batches flushes so packet-level diagnostics cannot become
    # the source of the audio stalls being measured.
    if _write_trace_line(line):
        _trace_dropped = 0
    else:
        _trace_dropped += 1


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


class RateLimiter:
    """First call always fires; then at most once per *period_s*."""

    def __init__(self, period_s: float = 1.0) -> None:
        self.period_s = hires_period(period_s)
        self._last = 0.0
        self._first = True

    def due(self) -> bool:
        now = time.monotonic()
        if self._first or (now - self._last) >= self.period_s:
            self._first = False
            self._last = now
            return True
        return False

    def reset(self) -> None:
        self._first = True
        self._last = 0.0


class StreamMeter:
    """Accumulate a 1 Hz SA_TRACE for a high-frequency stream hop.

    Numeric fields are summed except known peak fields (rms, qsize, …) which
    take the window max. Non-numeric fields keep the last value.
    """

    def __init__(self, agent: str, event: str, period_s: float | None = None, **fixed: Any) -> None:
        self.agent = agent
        self.event = event
        self.period_s = STREAM_PERIOD_S if period_s is None else period_s
        self.fixed = fixed
        self._t0 = time.monotonic()
        self._n = 0
        self._sum: dict[str, float] = {}
        self._max: dict[str, float] = {}
        self._last: dict[str, Any] = {}

    def add(self, **fields: Any) -> None:
        if not _STREAM:
            return
        self._n += 1
        for key, value in fields.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                self._last[key] = value
                continue
            number = float(value)
            if key in _METER_LAST_KEYS:
                self._last[key] = int(number) if number == int(number) else number
            elif key in _METER_MAX_KEYS:
                prev = self._max.get(key)
                self._max[key] = number if prev is None else max(prev, number)
            else:
                self._sum[key] = self._sum.get(key, 0.0) + number
        if (time.monotonic() - self._t0) >= self.period_s:
            self.flush()

    def flush(self) -> None:
        if not _STREAM:
            self._clear()
            return
        if self._n <= 0 and not self._last and not self.fixed:
            self._t0 = time.monotonic()
            return
        window = max(time.monotonic() - self._t0, 1e-6)
        payload: dict[str, Any] = dict(self.fixed)
        payload.update(self._last)
        for key, value in self._sum.items():
            payload[key] = int(value) if value == int(value) else round(value, 3)
        for key, value in self._max.items():
            payload[key] = int(value) if value == int(value) else round(value, 3)
        payload["n"] = self._n
        payload["window_s"] = round(window, 3)
        trace_stream(self.agent, self.event, **payload)
        self._clear()

    def _clear(self) -> None:
        self._n = 0
        self._sum.clear()
        self._max.clear()
        self._last.clear()
        self._t0 = time.monotonic()


class TypeCounter:
    """Roll up WebSocket / SSE event type names into one 1 Hz line."""

    def __init__(self, agent: str, event: str, period_s: float | None = None) -> None:
        self.agent = agent
        self.event = event
        self.period_s = STREAM_PERIOD_S if period_s is None else period_s
        self._t0 = time.monotonic()
        self._counts: dict[str, int] = {}

    def add(self, typ: str) -> None:
        if not _STREAM:
            return
        name = (typ or "empty")[:80]
        self._counts[name] = self._counts.get(name, 0) + 1
        if (time.monotonic() - self._t0) >= self.period_s:
            self.flush()

    def flush(self) -> None:
        if not self._counts:
            self._t0 = time.monotonic()
            return
        total = sum(self._counts.values())
        ranked = sorted(self._counts.items(), key=lambda kv: (-kv[1], kv[0]))[:16]
        types = ",".join(f"{name}:{count}" for name, count in ranked)
        trace_stream(
            self.agent,
            self.event,
            types=types,
            unique=len(self._counts),
            total=total,
            window_s=round(time.monotonic() - self._t0, 3),
        )
        self._counts.clear()
        self._t0 = time.monotonic()


class trace_turn:
    """Scopes a trace_id and emits ``<event>.start`` / ``<event>.end``."""

    def __init__(self, agent: str, event: str = "turn", trace_id: str = "", **fields: Any) -> None:
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
