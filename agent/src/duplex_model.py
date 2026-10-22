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
    return f"{base}/v1/realtime?duplex=1"


def _ref_audio_data_url() -> str | None:
    path = (os.environ.get("REF_AUDIO_PATH") or "").strip()
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        LOG.warning("REF_AUDIO_PATH missing: %s", path)
        return None
    raw = p.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


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
        self._last_append = 0.0
        self._append_bytes = 0
        self._append_n = 0
        self._video_pending: str | None = None
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
        url = _model_ws_url()
        model = os.environ.get("MODEL_NAME", "openbmb/MiniCPM-o-4_5")
        ref = _ref_audio_data_url()
        trace_stream("speech", "duplex.ws_open", url=url, model=model, has_ref=bool(ref))
        delay = 2.0
        last_err = "unknown"
        connected_at = 0
        for attempt in range(1, 181):
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
        session: dict[str, Any] = {
            "model": model,
            "modalities": ["audio", "text"],
            "instructions": self.system_prompt
            or "You are a helpful, friendly AI assistant. Respond naturally and conversationally.",
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": None,
            "overlap_policy": "listen_only",
            "playback_commit_policy": "ack_only",
            "temperature": 0.7,
            "idle_timeout_s": 300,
            "extra_body": {
                "auto_response": True,
                "force_listen_count": 0,
            },
        }
        if ref:
            session["ref_audio"] = ref
            trace_stream("speech", "voice.ref_loaded", source="REF_AUDIO_PATH")
        else:
            trace_stream("speech", "voice.ref_loaded", source="HT_ref_audio.wav")
        self._reader = asyncio.create_task(self._read())
        await self._send({"type": "session.update", "session": session})
        trace_stream(
            "speech",
            "duplex.session_update",
            modalities=session.get("modalities"),
            auto_response=True,
            overlap_policy=session.get("overlap_policy"),
            has_ref=bool(ref),
            instructions_len=len(str(session.get("instructions") or "")),
            chunk_ms=CHUNK_MS,
        )
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=30.0)
        except asyncio.TimeoutError as exc:
            raise RuntimeError("model-server never sent session.created") from exc
        self._hb = asyncio.create_task(self._heartbeat())

    async def _heartbeat(self) -> None:
        try:
            while self._ws is not None:
                await asyncio.sleep(20.0)
                try:
                    await self._send({"type": "session.heartbeat"})
                    now = time.monotonic()
                    trace_stream(
                        "speech",
                        "duplex.heartbeat",
                        rx_age_ms=_age_ms(self._t_last_rx, now),
                        delta_age_ms=_age_ms(self._t_last_delta, now),
                        speaking=self._speaking,
                    )
                    if self._speaking and self._t_last_delta and (now - self._t_last_delta) >= 2.0:
                        if self._stall_rl.due():
                            trace_stream(
                                "speech",
                                "stream.stall",
                                where="duplex.delta",
                                gap_ms=_age_ms(self._t_last_delta, now),
                                response_id=self._response_id[:16],
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

    async def _handle(self, ev: dict[str, Any]) -> None:
        typ = str(ev.get("type") or "")
        self._rx_types.add(typ)
        if typ in ("session.created", "session.updated"):
            sess = ev.get("session") if isinstance(ev.get("session"), dict) else {}
            sid = str(sess.get("id") or "")[:16]
            self._t_ready = time.monotonic()
            self._ready.set()
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
            trace_stream("speech", "error.duplex", error=_err_text(ev), type=typ, **summary)
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
        self._speaking = True
        self._delta_n += 1
        self._delta_bytes += len(pcm)
        self._turn_deltas += 1
        self._turn_bytes += len(pcm)
        self._t_last_delta = now
        if first:
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
                # Keep the older event name so existing Grafana panels still plot.
                trace_stream("speech", "duplex.speak", latency_ms=lat, bytes=len(pcm), first_audio=True)
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
                await self._append(piece, is_speech=False)
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

    async def _append(self, pcm16: bytes, is_speech: bool = True) -> None:
        if self._ws is None:
            return
        duration_ms = int(1000.0 * (len(pcm16) / 2.0) / MODEL_RATE)
        self._audio_end_ms += duration_ms
        video_attached = bool(self._video_pending)
        body: dict[str, Any] = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm16).decode("ascii"),
            "input_audio_format": "pcm16",
            "sample_rate_hz": MODEL_RATE,
            "duration_ms": duration_ms,
            "audio_end_ms": self._audio_end_ms,
            "is_speech": bool(is_speech),
        }
        if self._video_pending:
            # Bare JPEG base64 — a data: URL is rejected by the duplex runner.
            body["video_frames"] = [self._video_pending]
            self._video_pending = None
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
        """Flush leftover PCM and commit the user turn (VAD stop)."""
        if self._buf:
            piece = bytes(self._buf)
            self._buf.clear()
            await self._append(piece, is_speech=True)
        await self._commit_input()
        self._last_is_speech = False

    async def push_image(self, jpeg: bytes) -> None:
        if not jpeg:
            return
        if self._speaking:
            self._video_drop.add(dropped=1, reason="speaking")
            return
        if not self.ready:
            self._video_drop.add(dropped=1, reason="not_ready")
            return
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
        for meter in (self._rx_types, self._append_meter, self._delta_meter, self._drop_meter, self._video_drop):
            try:
                meter.flush()
            except Exception:
                pass
        for attr in ("_hb", "_reader"):
            task = getattr(self, attr)
            if task is not None:
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
                reason=self._close_reason,
                delta_n=self._delta_n,
                append_n=self._append_n,
                played_ms=self._played_ms,
            )
