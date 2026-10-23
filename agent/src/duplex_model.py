"""MiniCPM-o 4.5 full-duplex client over vLLM-Omni ``/v1/realtime?duplex=1``."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import websockets

from audio import MODEL_RATE, pcm_rms
from trace import RateLimiter, StreamMeter, TypeCounter, trace_exc, trace_stream

LOG = logging.getLogger("duplex")
CHUNK_MS = int(os.environ.get("AUDIO_CHUNK_MS", "100"))
# How long (ms) to tolerate no audio delta before cancelling + reconnecting.
# HiFT eager-fallback on vLLM-Omni 0.30.0 produces 5–30 s gaps between
# deltas on live turns; the vocoder DOES come back. Set high enough to ride
# those out, low enough to recover from true zombie sessions.
STALL_CANCEL_MS = int(os.environ.get("STALL_CANCEL_MS", "20000"))

_AUDIO_DELTA_TYPES = frozenset(
    {
        "response.audio.delta",
        "response.output_audio.delta",
        "audio.delta",
        "response.output_audio.delta.audio",
    }
)
_TEXT_DELTA_TYPES = frozenset(
    {
        "response.text.delta",
        "response.output_text.delta",
        "response.audio_transcript.delta",
        "response.output_audio_transcript.delta",
    }
)
_SPEECH_START_TYPES = frozenset(
    {
        "input_audio_buffer.speech_started",
        "conversation.item.input_audio.speech_started",
    }
)
_SPEECH_STOP_TYPES = frozenset(
    {
        "input_audio_buffer.speech_stopped",
        "conversation.item.input_audio.speech_stopped",
    }
)
_COMMIT_TYPES = frozenset(
    {
        "input_audio_buffer.committed",
        "input_audio_buffer.committed.completed",
    }
)
_QUIET_TYPES = frozenset(
    {
        "session.heartbeat_ack",
        "playback.acknowledged",
        "rate_limits.updated",
    }
)


def _model_ws_url() -> str:
    base = (os.environ.get("MODEL_BASE_URL") or "http://127.0.0.1:8199").rstrip("/")
    if base.startswith("http://"):
        base = "ws://" + base[len("http://") :]
    elif base.startswith("https://"):
        base = "wss://" + base[len("https://") :]
    # autostart=0: first message is session.update with ref_audio. v0.30.0
    # rejects a default session that has no ref_audio.
    return f"{base}/v1/realtime?duplex=1&autostart=0"


def _find_bundled_ref_audio() -> Path | None:
    hf_home = Path(os.environ.get("HF_HOME") or "/models/hf")
    roots = (
        hf_home / "hub" / "models--openbmb--MiniCPM-o-4_5",
        Path("/models/hf/hub/models--openbmb--MiniCPM-o-4_5"),
    )
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve() if root.exists() else root
        if resolved in seen or not root.is_dir():
            continue
        seen.add(resolved)
        matches = sorted(root.glob("snapshots/*/assets/HT_ref_audio.wav"))
        if matches:
            return matches[-1]
    fallback = Path("/models")
    if fallback.is_dir():
        hits = sorted(fallback.glob("**/assets/HT_ref_audio.wav"))
        if hits:
            return hits[-1]
    return None


def _ref_audio_data_url() -> tuple[str, str]:
    """Return (data_url, source_label). v0.30.0 duplex requires this for audio out."""
    override = (os.environ.get("REF_AUDIO_PATH") or "").strip()
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"REF_AUDIO_PATH missing: {path}")
        source = "REF_AUDIO_PATH"
    else:
        path = _find_bundled_ref_audio()
        if path is None:
            raise FileNotFoundError(
                "MiniCPM-o duplex requires ref_audio. Set REF_AUDIO_PATH to a "
                "16 kHz mono WAV, or keep the bundled assets/HT_ref_audio.wav "
                "in the HuggingFace snapshot under /models/hf."
            )
        source = str(path)
    raw = path.read_bytes()
    if len(raw) < 64 or raw[:4] != b"RIFF":
        raise ValueError(f"ref_audio is not a WAV file: {path}")
    return f"data:audio/wav;base64,{base64.b64encode(raw).decode('ascii')}", source


def _err_text(ev: dict[str, Any]) -> str:
    err = ev.get("error")
    if isinstance(err, dict):
        return str(err.get("message") or err.get("code") or err)[:400]
    return str(err or ev)[:400]


_SUMMARY_SKIP = frozenset({"type", "error", "event_id", "audio", "delta", "video_frames"})


def _ev_summary(ev: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in list(ev.items())[:16]:
        if key in _SUMMARY_SKIP:
            if key in {"audio", "delta", "video_frames"} and isinstance(value, (str, list)):
                out[f"{key}_len"] = len(value)
            continue
        if isinstance(value, dict):
            out[key] = "{" + ",".join(list(value.keys())[:8]) + "}"
        elif isinstance(value, list):
            out[key] = f"list:{len(value)}"
        else:
            out[key] = str(value)[:80]
    return out


def _age_ms(stamp: float, now: float) -> int:
    if stamp <= 0:
        return -1
    return int((now - stamp) * 1000)


class MiniCPMDuplexEngine:
    """Stream 16 kHz PCM in, receive 24 kHz PCM out. Model decides listen/speak."""

    input_rate = MODEL_RATE

    def __init__(self, system_prompt: str = "") -> None:
        self.system_prompt = system_prompt.strip()
        self._ws: Any = None
        self._out: asyncio.Queue[tuple[bytes, int]] = asyncio.Queue()
        self._reader: asyncio.Task | None = None
        self._hb: asyncio.Task | None = None
        self._buf = bytearray()
        self._in_resampler = None
        self._speaking = False
        self._ready = asyncio.Event()
        self._handshake = asyncio.Event()
        self._handshake_error = ""
        self._last_append = 0.0
        self._append_bytes = 0
        self._append_n = 0
        self._video_pending: str | None = None
        self._video_n = 0
        self._last_video_ms = -1_000
        self._played_ms = 0
        self._audio_end_ms = 0
        self._evt = 0
        self._response_id = ""
        self._t_speak = 0.0
        self._t_user_speech = 0.0
        self._t_ready = 0.0
        self._t_last_mic = 0.0
        self._t_last_append = 0.0
        self._t_last_delta = 0.0
        self._t_last_rx = 0.0
        self._out_rate = 24000
        self._last_is_speech = False
        self._commit_pending = False
        self._delta_n = 0
        self._delta_bytes = 0
        self._turn_deltas = 0
        self._turn_bytes = 0
        self._turn_audio_ms = 0
        self._turn_max_gap_ms = -1
        self._turn_gen_t0 = 0.0
        self._last_rtf = 0.0
        self._delta_timing_rl = RateLimiter(2.0)
        self._text_n = 0
        self._bin_n = 0
        self._acks_tx = 0
        self._acks_rx = 0
        self._close_code = 0
        self._close_reason = ""
        self._rx_types = TypeCounter("speech", "duplex.rx_stats")
        self._append_meter = StreamMeter("speech", "duplex.append", kind="in")
        self._delta_meter = StreamMeter("speech", "duplex.audio_delta", kind="out")
        self._drop_meter = StreamMeter("speech", "stream.drop_not_ready", reason="ws_not_ready")
        self._video_drop = StreamMeter("speech", "stream.video_drop")
        self._text_rl = RateLimiter(1.0)
        self._unknown_rl = RateLimiter(2.0)
        self._stall_rl = RateLimiter(2.0)
        self._watchdog: asyncio.Task | None = None
        self._reconnect_reason = ""
        self._last_reconnect = 0.0
        self._closing = False

    @property
    def generating(self) -> bool:
        return self._speaking

    @property
    def ready(self) -> bool:
        return self._ws is not None and self._ready.is_set()

    def stream_snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "ws_open": self._ws is not None,
            "ready": self._ready.is_set(),
            "speaking": self._speaking,
            "out_q": self._out.qsize(),
            "buf_ms": int(1000.0 * (len(self._buf) / 2.0) / MODEL_RATE) if self._buf else 0,
            "audio_end_ms": self._audio_end_ms,
            "played_ms": self._played_ms,
            "delta_n": self._delta_n,
            "append_n": self._append_n,
            "mic_age_ms": _age_ms(self._t_last_mic, now),
            "append_age_ms": _age_ms(self._t_last_append, now),
            "delta_age_ms": _age_ms(self._t_last_delta, now),
            "rx_age_ms": _age_ms(self._t_last_rx, now),
            "acks_tx": self._acks_tx,
            "acks_rx": self._acks_rx,
            "video_n": self._video_n,
            "video_pending": bool(self._video_pending),
        }

    def _eid(self) -> str:
        self._evt += 1
        return f"evt_{self._evt:04d}"

    async def _send(self, body: dict[str, Any]) -> None:
        if self._ws is None:
            return
        body.setdefault("event_id", self._eid())
        await self._ws.send(json.dumps(body))

    async def start(self) -> None:
        self._closing = False
        await self._connect_and_handshake()
        if self._watchdog is None or self._watchdog.done():
            self._watchdog = asyncio.create_task(self._watchdog_loop())

    async def _connect_and_handshake(self) -> None:
        url = _model_ws_url()
        model = os.environ.get("MODEL_NAME", "openbmb/MiniCPM-o-4_5")
        ref, ref_source = _ref_audio_data_url()
        instructions = self.system_prompt or (
            "Clone the voice in the provided audio prompt. Please assist users "
            "while maintaining this voice style. Please answer the user's questions "
            "seriously and in a high quality. Please chat with the user in a highly "
            "human-like and oral style. Always reply in English. Never switch to "
            "Chinese unless the user is speaking Chinese. You are a helpful assistant."
        )
        trace_stream("speech", "duplex.ws_open", url=url, model=model, has_ref=True, ref_source=ref_source[-80:])
        delay = 2.0
        last_err = "unknown"
        connected_at = 0
        for attempt in range(1, 181):
            if self._closing:
                raise RuntimeError("duplex closed before connect")
            try:
                self._ws = await websockets.connect(url, max_size=16_000_000, ping_interval=20)
                connected_at = attempt
                break
            except Exception as exc:
                last_err = str(exc)[:200]
                trace_stream("speech", "duplex.ws_retry", attempt=attempt, error=last_err, url=url)
                await asyncio.sleep(min(delay, 15.0))
                delay = min(delay * 1.25, 15.0)
        else:
            raise RuntimeError(f"model-server duplex never accepted {url}: {last_err}")

        trace_stream(
            "speech",
            "duplex.ws_connected",
            url=url,
            attempt=connected_at,
            model=model,
        )
        # v0.30.0 defaults sliding_window_mode=off and high=8000, both of which
        # overflow our stage-0 max_model_len. basic 3000/2000 trims units
        # (keeps the system/ref-audio prefix) before the scheduler rejects.
        session: dict[str, Any] = {
            "model": model,
            "modalities": ["audio", "text"],
            "instructions": instructions,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": None,
            "overlap_policy": "listen_only",
            "playback_commit_policy": "ack_only",
            "temperature": 0.7,
            "idle_timeout_s": 300,
            "ref_audio": ref,
            "extra_body": {
                "auto_response": True,
                "force_listen_count": 0,
                "ref_audio": ref,
                "sliding_window_mode": "basic",
                "basic_window_high_tokens": 3000,
                "basic_window_low_tokens": 2000,
            },
        }
        trace_stream("speech", "voice.ref_loaded", source=ref_source[-120:], bytes=len(ref))
        self._handshake.clear()
        self._handshake_error = ""
        self._ready.clear()
        self._reader = asyncio.create_task(self._read())
        await self._send({"type": "session.update", "session": session})
        trace_stream(
            "speech",
            "duplex.session_update",
            modalities=session.get("modalities"),
            auto_response=True,
            overlap_policy=session.get("overlap_policy"),
            has_ref=True,
            sliding_window="basic",
            window_high=3000,
            window_low=2000,
            instructions_len=len(instructions),
            chunk_ms=CHUNK_MS,
        )
        try:
            await asyncio.wait_for(self._handshake.wait(), timeout=30.0)
        except asyncio.TimeoutError as exc:
            raise RuntimeError("model-server never sent session.created") from exc
        if self._handshake_error:
            raise RuntimeError(f"duplex handshake failed: {self._handshake_error}")
        if not self._ready.is_set():
            raise RuntimeError("model-server never sent session.created")
        self._hb = asyncio.create_task(self._heartbeat())

    def _request_reconnect(self, reason: str) -> None:
        if self._closing or self._reconnect_reason:
            return
        self._reconnect_reason = reason[:120]
        trace_stream("speech", "duplex.reconnect_request", reason=self._reconnect_reason)

    async def _watchdog_loop(self) -> None:
        try:
            while not self._closing:
                await asyncio.sleep(0.5)
                reason = self._reconnect_reason
                if not reason:
                    continue
                now = time.monotonic()
                if now - self._last_reconnect < 15.0:
                    self._reconnect_reason = ""
                    continue
                self._reconnect_reason = ""
                self._last_reconnect = now
                try:
                    await self._reconnect(reason)
                except Exception as exc:
                    trace_exc("speech", "error.duplex_reconnect", exc, reason=reason)
        except asyncio.CancelledError:
            raise

    async def _reconnect(self, reason: str) -> None:
        trace_stream("speech", "duplex.reconnect", reason=reason)
        self._ready.clear()
        self._speaking = False
        await self._teardown_socket(cancel_watchdog=False)
        self._reset_stream_state()
        await self._connect_and_handshake()
        trace_stream("speech", "duplex.reconnected", reason=reason)

    def _reset_stream_state(self) -> None:
        self._buf.clear()
        self._audio_end_ms = 0
        self._played_ms = 0
        self._last_video_ms = -1_000
        self._commit_pending = False
        self._last_is_speech = False
        self._turn_deltas = 0
        self._turn_bytes = 0
        self._response_id = ""
        self._t_last_delta = 0.0
        self._t_speak = 0.0
        while not self._out.empty():
            try:
                self._out.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _teardown_socket(self, *, cancel_watchdog: bool) -> None:
        tasks = ["_hb", "_reader"]
        if cancel_watchdog:
            tasks.append("_watchdog")
        current = asyncio.current_task()
        for attr in tasks:
            task = getattr(self, attr)
            if task is None or task is current or task.done():
                setattr(self, attr, None)
                continue
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            setattr(self, attr, None)
        if self._ws is not None:
            try:
                await self._send({"type": "session.close"})
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
            trace_stream(
                "speech",
                "duplex.ws_close",
                code=self._close_code,
                reason=self._close_reason or "teardown",
                delta_n=self._delta_n,
                append_n=self._append_n,
                played_ms=self._played_ms,
            )

    async def _heartbeat(self) -> None:
        ticks = 0
        try:
            while self._ws is not None:
                await asyncio.sleep(2.0)
                ticks += 1
                try:
                    now = time.monotonic()
                    stalled = (
                        self._speaking
                        and self._turn_deltas > 0
                        and self._t_last_delta > 0.0
                        and (now - self._t_last_delta) >= 2.0
                    )
                    if stalled and self._stall_rl.due():
                        gap = _age_ms(self._t_last_delta, now)
                        trace_stream(
                            "speech",
                            "stream.stall",
                            where="duplex.delta",
                            gap_ms=gap,
                            response_id=self._response_id[:16],
                        )
                        # Hung mid-utterance after context overflow: the server
                        # drops the error and leaves a zombie session that then
                        # speaks Chinese (system prefix gone). Reopen.
                        # HiFT eager-fallback gaps are 5–30 s; only cancel
                        # after STALL_CANCEL_MS to avoid chopping real speech.
                        if gap >= STALL_CANCEL_MS:
                            trace_stream(
                                "speech",
                                "duplex.stall_cancel",
                                gap_ms=gap,
                                response_id=self._response_id[:16],
                            )
                            await self.interrupt()
                            self._request_reconnect(f"stall_{gap}ms")
                    if ticks % 10 == 0:
                        await self._send({"type": "session.heartbeat"})
                        trace_stream(
                            "speech",
                            "duplex.heartbeat",
                            rx_age_ms=_age_ms(self._t_last_rx, now),
                            delta_age_ms=_age_ms(self._t_last_delta, now),
                            speaking=self._speaking,
                        )
                except Exception:
                    return
        except asyncio.CancelledError:
            raise

    async def _read(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                self._t_last_rx = time.monotonic()
                if isinstance(raw, bytes):
                    self._bin_n += 1
                    if raw:
                        if self._bin_n == 1:
                            trace_stream("speech", "duplex.binary_frame", bytes=len(raw), n=1)
                        await self._out.put((raw, self._out_rate))
                    continue
                try:
                    ev = json.loads(raw)
                except json.JSONDecodeError:
                    trace_stream("speech", "duplex.bad_json", bytes=len(raw) if isinstance(raw, str) else 0)
                    continue
                await self._handle(ev)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            code = getattr(exc, "code", None)
            reason = str(getattr(exc, "reason", "") or exc)[:200]
            self._close_code = int(code) if isinstance(code, int) else 0
            self._close_reason = reason
            trace_exc(
                "speech",
                "error.duplex_ws",
                exc,
                code=self._close_code,
                reason=reason,
                ready=self._ready.is_set(),
                speaking=self._speaking,
            )
            if not self._ready.is_set() and not self._handshake.is_set():
                self._handshake_error = reason or "duplex websocket closed"
                self._handshake.set()
            elif self._ready.is_set() and not self._closing:
                self._request_reconnect(reason or "duplex_ws_closed")
        else:
            if not self._ready.is_set() and not self._handshake.is_set():
                self._handshake_error = self._handshake_error or "duplex websocket closed before session.created"
                self._handshake.set()
            elif self._ready.is_set() and not self._closing:
                self._request_reconnect("duplex_ws_closed")

    async def _handle(self, ev: dict[str, Any]) -> None:
        typ = str(ev.get("type") or "")
        self._rx_types.add(typ)
        if typ in ("session.created", "session.updated"):
            sess = ev.get("session") if isinstance(ev.get("session"), dict) else {}
            sid = str(sess.get("id") or "")[:16]
            self._t_ready = time.monotonic()
            self._ready.set()
            self._handshake.set()
            trace_stream(
                "speech",
                "duplex.session_created" if typ.endswith("created") else "duplex.session_updated",
                session_id=sid,
                keys=",".join(list(sess.keys())[:12]),
            )
            return
        if typ == "session.heartbeat_ack":
            self._acks_rx += 1
            return
        if typ in ("response.created", "response.speak"):
            self._response_id = str(ev.get("response_id") or (ev.get("response") or {}).get("id") or "")
            self._t_speak = time.monotonic()
            self._turn_deltas = 0
            self._turn_bytes = 0
            self._turn_audio_ms = 0
            self._turn_max_gap_ms = -1
            self._turn_gen_t0 = 0.0
            if typ == "response.speak":
                self._speaking = True
                from_speech = _age_ms(self._t_user_speech, self._t_speak)
                trace_stream(
                    "speech",
                    "duplex.speak",
                    response_id=self._response_id[:16],
                    from_speech_ms=from_speech,
                )
            else:
                trace_stream("speech", "duplex.response_created", response_id=self._response_id[:16])
            return
        if typ in _AUDIO_DELTA_TYPES:
            await self._on_audio_delta(ev)
            return
        if typ in _TEXT_DELTA_TYPES:
            self._text_n += 1
            text = str(ev.get("delta") or ev.get("text") or "")[:80]
            if self._text_n == 1 or self._text_rl.due():
                trace_stream("speech", "duplex.text_delta", n=self._text_n, text=text, chars=len(text))
            return
        if typ in ("response.listen",) or ev.get("decision") == "listen":
            trace_stream(
                "speech",
                "duplex.listen",
                response_id=self._response_id[:16],
                turn_bytes=self._turn_bytes,
                turn_deltas=self._turn_deltas,
                played_ms=self._played_ms,
            )
            self._speaking = False
            return
        if typ in ("response.audio.done", "response.output_audio.done"):
            trace_stream(
                "speech",
                "duplex.audio_done",
                response_id=self._response_id[:16],
                turn_bytes=self._turn_bytes,
                turn_deltas=self._turn_deltas,
            )
            return
        if typ in ("response.done", "response.completed"):
            status = str((ev.get("response") or {}).get("status") or "")
            speak_ms = _age_ms(self._t_speak, time.monotonic()) if self._t_speak else -1
            trace_stream(
                "speech",
                "duplex.response_done",
                status=status,
                played_ms=self._played_ms,
                turn_bytes=self._turn_bytes,
                turn_deltas=self._turn_deltas,
                speak_ms=speak_ms,
                response_id=self._response_id[:16],
            )
            self._speaking = False
            return
        if typ in ("error", "response.error"):
            summary = _ev_summary(ev)
            summary.pop("type", None)
            summary.pop("error", None)
            err = _err_text(ev)
            trace_stream("speech", "error.duplex", error=err, type=typ, **summary)
            if not self._ready.is_set():
                self._handshake_error = err
                self._handshake.set()
                return
            low = err.lower()
            if "context_length" in low or "max_model_len" in low:
                self._request_reconnect(err[:80])
            return
        if typ in _SPEECH_START_TYPES:
            trace_stream("speech", "duplex.model_speech_start", ev_type=typ, keys=",".join(list(ev.keys())[:12]))
            return
        if typ in _SPEECH_STOP_TYPES:
            trace_stream("speech", "duplex.model_speech_stop", **_ev_summary(ev))
            return
        if typ in _COMMIT_TYPES:
            trace_stream("speech", "duplex.committed", audio_end_ms=self._audio_end_ms)
            return
        if typ == "playback.acknowledged":
            self._acks_rx += 1
            return
        if typ == "rate_limits.updated":
            limits = ev.get("rate_limits") or ev.get("limits") or {}
            trace_stream("speech", "duplex.rate_limits", **_ev_summary(limits if isinstance(limits, dict) else ev))
            return
        if typ.startswith("input_audio_buffer."):
            if self._unknown_rl.due():
                trace_stream("speech", "duplex.event", type=typ, **_ev_summary(ev))
            return
        if typ in _QUIET_TYPES:
            return
        if self._unknown_rl.due() or not typ:
            trace_stream("speech", "duplex.event", type=typ or "empty", **_ev_summary(ev))

    async def _on_audio_delta(self, ev: dict[str, Any]) -> None:
        b64 = ev.get("delta") or ev.get("audio") or ""
        if not b64:
            return
        try:
            pcm = base64.b64decode(b64)
        except Exception:
            trace_stream("speech", "duplex.bad_audio_b64", chars=len(str(b64)))
            return
        if pcm[:4] == b"RIFF":
            pcm = pcm[44:]
        now = time.monotonic()
        first = (not self._speaking) or self._turn_deltas == 0
        # Inter-delta gap: time the ring sat waiting for this delta.
        gap_ms = -1 if first else _age_ms(self._t_last_delta, now)
        dur_ms = int(1000.0 * (len(pcm) / 2.0) / max(self._out_rate, 1))
        self._speaking = True
        self._delta_n += 1
        self._delta_bytes += len(pcm)
        self._turn_deltas += 1
        self._turn_bytes += len(pcm)
        self._turn_audio_ms += dur_ms
        if gap_ms > self._turn_max_gap_ms:
            self._turn_max_gap_ms = gap_ms
        self._t_last_delta = now
        if first:
            self._turn_gen_t0 = now
            lat = _age_ms(self._t_speak, now) if self._t_speak else -1
            from_speech = _age_ms(self._t_user_speech, now)
            from_ready = _age_ms(self._t_ready, now)
            trace_stream(
                "speech",
                "duplex.first_audio",
                latency_ms=lat,
                from_speech_ms=from_speech,
                from_ready_ms=from_ready,
                bytes=len(pcm),
                rate=self._out_rate,
                response_id=self._response_id[:16],
                qsize=self._out.qsize() + 1,
            )
            if lat >= 0:
                trace_stream("speech", "duplex.speak", latency_ms=lat, bytes=len(pcm), first_audio=True)
        # RTF = audio produced / wall time since first delta.  RTF < 1 means
        # the model emits slower than real time — underruns are fundamental.
        gen_wall_ms = _age_ms(self._turn_gen_t0, now) if self._turn_gen_t0 else 0
        rtf = round(self._turn_audio_ms / gen_wall_ms, 2) if gen_wall_ms > 0 else 0.0
        self._last_rtf = rtf
        if self._turn_deltas <= 5 or gap_ms >= 250 or self._delta_timing_rl.due():
            trace_stream(
                "speech",
                "duplex.delta_timing",
                gap_ms=gap_ms,
                dur_ms=dur_ms,
                bytes=len(pcm),
                qsize=self._out.qsize() + 1,
                turn_audio_ms=self._turn_audio_ms,
                gen_wall_ms=gen_wall_ms,
                rtf=rtf,
                deltas=self._turn_deltas,
                response_id=self._response_id[:16],
            )
        self._delta_meter.add(deltas=1, bytes=len(pcm), qsize=self._out.qsize() + 1, rms=pcm_rms(pcm))
        await self._out.put((pcm, self._out_rate))

    async def push_audio(self, pcm16: bytes, sample_rate: int, is_speech: bool = True) -> None:
        from audio import LinearResampler

        if not pcm16:
            return
        self._t_last_mic = time.monotonic()
        if self._ws is None or not self._ready.is_set():
            self._drop_meter.add(bytes=len(pcm16), frames=1)
            return
        if sample_rate != MODEL_RATE:
            if self._in_resampler is None:
                self._in_resampler = LinearResampler(MODEL_RATE)
            pcm16 = self._in_resampler.feed(pcm16, sample_rate)
            if not pcm16:
                return
        # While the talker is occupying the Ada, silence+JPEGs still grow
        # stage-0 KV and compete with generation. Barge-in still flows
        # because is_speech=True. Drop leftover silence so it is not dumped
        # the moment the model yields.
        if not is_speech and self._speaking:
            self._buf.clear()
            return
        self._buf.extend(pcm16)
        chunk_bytes = int(MODEL_RATE * (CHUNK_MS / 1000.0) * 2)
        while len(self._buf) >= chunk_bytes:
            piece = bytes(self._buf[:chunk_bytes])
            del self._buf[:chunk_bytes]
            await self._append(piece, is_speech=is_speech)
        # Speech ended: flush a leftover short chunk, then commit so the
        # server closes the user turn (vllm-omni reference client always
        # sends input_audio_buffer.commit after the last append).
        if self._last_is_speech and not is_speech:
            if self._buf:
                piece = bytes(self._buf)
                self._buf.clear()
                await self._append(piece, is_speech=False, force_video=True)
            else:
                await self._commit_input()

    async def _commit_input(self) -> None:
        if self._ws is None or not self._ready.is_set():
            return
        if not self._commit_pending:
            return
        self._commit_pending = False
        await self._send({"type": "input_audio_buffer.commit"})
        trace_stream("speech", "duplex.commit", audio_end_ms=self._audio_end_ms)

    async def _append(self, pcm16: bytes, is_speech: bool = True, *, force_video: bool = False) -> None:
        if self._ws is None:
            return
        duration_ms = int(1000.0 * (len(pcm16) / 2.0) / MODEL_RATE)
        self._audio_end_ms += duration_ms
        video_attached = False
        body: dict[str, Any] = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm16).decode("ascii"),
            "input_audio_format": "pcm16",
            "sample_rate_hz": MODEL_RATE,
            "duration_ms": duration_ms,
            "audio_end_ms": self._audio_end_ms,
            "is_speech": bool(is_speech),
        }
        # Native duplex expects ~1 JPEG per 1 s audio unit. Attach the latest
        # camera still on that cadence so vision rides the live mic stream.
        if self._video_pending and (
            force_video or (self._audio_end_ms - self._last_video_ms) >= 900
        ):
            # Bare JPEG base64 — a data: URL is rejected by the duplex runner.
            body["video_frames"] = [self._video_pending]
            self._last_video_ms = self._audio_end_ms
            self._video_n += 1
            video_attached = True
            jpeg_chars = len(self._video_pending)
            self._video_pending = None
            trace_stream(
                "speech",
                "duplex.video_attach",
                n=self._video_n,
                audio_end_ms=self._audio_end_ms,
                b64_chars=jpeg_chars,
                is_speech=int(bool(is_speech)),
            )
        await self._send(body)
        now = time.monotonic()
        self._append_n += 1
        self._append_bytes += len(pcm16)
        self._t_last_append = now
        rms = pcm_rms(pcm16)
        if is_speech and not self._last_is_speech:
            self._t_user_speech = now
            self._commit_pending = True
            trace_stream(
                "speech",
                "duplex.append_speech",
                rms=rms,
                duration_ms=duration_ms,
                audio_end_ms=self._audio_end_ms,
                video_attached=video_attached,
            )
        speech_ended = self._last_is_speech and not is_speech
        self._last_is_speech = bool(is_speech)
        if speech_ended:
            await self._commit_input()
        self._append_meter.add(
            chunks=1,
            bytes=len(pcm16),
            rms=rms,
            duration_ms=duration_ms,
            is_speech=int(bool(is_speech)),
            video_attached=int(video_attached),
        )

    async def end_of_speech(self) -> None:
        """Flush leftover PCM (with latest camera still) and commit the turn."""
        if self._buf:
            piece = bytes(self._buf)
            self._buf.clear()
            await self._append(piece, is_speech=True, force_video=True)
        await self._commit_input()
        self._last_is_speech = False

    async def push_image(self, jpeg: bytes) -> None:
        if not jpeg:
            return
        if not self.ready:
            self._video_drop.add(dropped=1, reason="not_ready")
            return
        # Keep the newest still even while the model is speaking so the next
        # 1 s audio unit still carries live camera (duplex in, speech out).
        self._video_pending = base64.b64encode(jpeg).decode("ascii")

    async def pull_audio(self) -> list[tuple[bytes, int]]:
        out: list[tuple[bytes, int]] = []
        while True:
            try:
                pcm, rate = self._out.get_nowait()
            except asyncio.QueueEmpty:
                break
            out.append((pcm, rate))
            dur_ms = int(1000.0 * (len(pcm) / 2.0) / max(rate, 1))
            self._played_ms += dur_ms
        if out and self._ws is not None and self._played_ms:
            try:
                await self._send(
                    {
                        "type": "playback.ack",
                        "played_ms": self._played_ms,
                        "committed_ms": self._played_ms,
                        "response_id": self._response_id,
                    }
                )
                self._acks_tx += 1
            except Exception:
                pass
        return out

    async def interrupt(self) -> None:
        was = self._speaking
        q = self._out.qsize()
        self._speaking = False
        while not self._out.empty():
            try:
                self._out.get_nowait()
            except asyncio.QueueEmpty:
                break
        if self._ws is None:
            return
        try:
            await self._send({"type": "response.cancel"})
            trace_stream(
                "speech",
                "duplex.cancel",
                was_speaking=was,
                dropped_q=q,
                response_id=self._response_id[:16],
            )
        except Exception as exc:
            trace_exc("speech", "error.duplex_cancel", exc)

    async def close(self) -> None:
        self._closing = True
        self._reconnect_reason = ""
        for meter in (self._rx_types, self._append_meter, self._delta_meter, self._drop_meter, self._video_drop):
            try:
                meter.flush()
            except Exception:
                pass
        await self._teardown_socket(cancel_watchdog=True)
        self._ready.clear()
