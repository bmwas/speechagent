"""Qwen2.5-Omni-3B live video+speech via vLLM-Omni ``/v1/video/chat/stream``.

Camera and mic go to the server as they happen (``video.frame`` +
``audio.chunk``). VAD stop submits ``video.query``; speech streams back on
``response.output_audio.delta``. This is not ``image_url`` and not a
post-hoc MP4. MiniCPM duplex stays in ``duplex_model.py``.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import time
import urllib.error
import urllib.request
import wave
from typing import Any
from urllib.parse import urlparse

import websockets

from audio import MODEL_RATE, LinearResampler
from trace import RateLimiter, StreamMeter, TypeCounter, trace_exc, trace_stream

LOG = logging.getLogger("qwen")

_QUERY_TEXT = "Reply in spoken English. Keep it concise."
_MIN_UTT_S = 0.2
_OUT_RATE = 24000
_QWEN_SYSTEM_PROMPT = (
    "You can see the user through their camera and hear them through their "
    "microphone. Always reply in spoken English. Keep responses concise."
)


def _system_prompt(raw: str) -> str:
    """Drop the MiniCPM clone prompt; Qwen has no reference WAV."""
    text = (raw or "").strip()
    if not text or "MiniCPM" in text or "Clone the voice" in text:
        return _QWEN_SYSTEM_PROMPT
    return text


def _stall_s() -> float:
    raw = (os.environ.get("QWEN_STALL_S") or "").strip()
    if not raw:
        return 12.0
    try:
        return max(4.0, float(raw))
    except ValueError:
        return 12.0


_AUDIO_DELTA_TYPES = frozenset(
    {
        "response.output_audio.delta",
        "response.audio.delta",
        "audio.delta",
    }
)
_TEXT_DELTA_TYPES = frozenset(
    {
        "response.text.delta",
        "response.output_text.delta",
    }
)
_DONE_TYPES = frozenset(
    {
        "response.output_audio.done",
        "response.audio.done",
        "response.done",
    }
)


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _model_http_url() -> str:
    base = (os.environ.get("MODEL_BASE_URL") or "http://127.0.0.1:8199").rstrip("/")
    if not base.startswith(("http://", "https://")):
        base = "http://" + base
    return base


def _model_ws_url() -> str:
    base = _model_http_url()
    if base.startswith("http://"):
        base = "ws://" + base[len("http://") :]
    elif base.startswith("https://"):
        base = "wss://" + base[len("https://") :]
    return f"{base}/v1/video/chat/stream"


def _model_host_port() -> tuple[str, int]:
    parsed = urlparse(_model_http_url())
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 8199)
    return host, port


async def probe_model_server() -> dict[str, Any]:
    """TCP + HTTP probe for vLLM-Omni (used while the WS endpoint is down)."""
    host, port = _model_host_port()
    out: dict[str, Any] = {
        "model_base_url": _model_http_url(),
        "host": host,
        "port": port,
        "tcp_ok": False,
        "http_ok": False,
        "models_ok": False,
        "detail": "",
    }
    try:
        _reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), 2.0)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        out["tcp_ok"] = True
    except Exception as exc:
        out["detail"] = f"tcp refused: {exc}"[:200]
        return out

    def _http_probe() -> tuple[bool, bool, str]:
        health_url = f"http://{host}:{port}/health"
        models_url = f"http://{host}:{port}/v1/models"
        detail_parts: list[str] = []
        http_ok = False
        models_ok = False
        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:
                body = resp.read(240).decode("utf-8", "replace")
                http_ok = 200 <= int(getattr(resp, "status", 200)) < 300
                detail_parts.append(f"health={body[:120]}")
        except urllib.error.HTTPError as exc:
            detail_parts.append(f"health HTTP {exc.code}")
        except Exception as exc:
            detail_parts.append(f"health {exc}")
        try:
            with urllib.request.urlopen(models_url, timeout=2) as resp:
                body = resp.read(240).decode("utf-8", "replace")
                models_ok = 200 <= int(getattr(resp, "status", 200)) < 300
                detail_parts.append(f"models={body[:120]}")
        except urllib.error.HTTPError as exc:
            detail_parts.append(f"models HTTP {exc.code}")
        except Exception as exc:
            detail_parts.append(f"models {exc}")
        return http_ok, models_ok, "; ".join(detail_parts)[:240]

    try:
        http_ok, models_ok, detail = await asyncio.to_thread(_http_probe)
        out["http_ok"] = http_ok
        out["models_ok"] = models_ok
        out["detail"] = detail
    except Exception as exc:
        out["detail"] = f"http probe failed: {exc}"[:200]
    return out


def _age_ms(started: float, now: float) -> int:
    if not started:
        return -1
    return int((now - started) * 1000)


def _pcm_from_delta(ev: dict[str, Any]) -> tuple[bytes, int]:
    encoded = ev.get("data") or ev.get("delta") or ev.get("audio") or ""
    if not encoded or not isinstance(encoded, str):
        return b"", _OUT_RATE
    try:
        raw = base64.b64decode(encoded)
    except Exception:
        return b"", _OUT_RATE
    if raw[:4] == b"RIFF":
        try:
            with wave.open(io.BytesIO(raw), "rb") as w:
                return w.readframes(w.getnframes()), int(w.getframerate() or _OUT_RATE)
        except Exception:
            return raw[44:], _OUT_RATE
    return raw, _OUT_RATE


class QwenOmniEngine:
    """Persistent video-stream session. Same duck-type SpeechSession already calls."""

    input_rate = MODEL_RATE

    def __init__(self, system_prompt: str = "") -> None:
        self.system_prompt = _system_prompt(system_prompt)
        self.speaker = os.environ.get("QWEN_SPEAKER", "Chelsie").strip() or "Chelsie"
        self._stall_s = _stall_s()
        self._resetting = False
        self.model = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-Omni-3B")
        self._max_frames = max(1, min(_env_int("QWEN_VIDEO_MAX_FRAMES", 50), 256))
        self._num_frames = max(1, min(_env_int("QWEN_VIDEO_NUM_FRAMES", 40), 128))
        if self._num_frames > self._max_frames:
            self._num_frames = self._max_frames
        self._ws: Any = None
        self._out: asyncio.Queue[tuple[bytes, int]] = asyncio.Queue()
        self._in_rs: LinearResampler | None = None
        self._ready = asyncio.Event()
        self._busy = False
        self._drop_output = False
        self._reader: asyncio.Task | None = None
        self._hb: asyncio.Task | None = None
        self._frame_n = 0
        self._audio_n = 0
        self._utt_bytes = 0
        self._t_last_mic = 0.0
        self._t_last_frame = 0.0
        self._t_last_delta = 0.0
        self._t_last_rx = 0.0
        self._t_query = 0.0
        self._t_ready = 0.0
        self._first_audio = False
        self._text_n = 0
        self._text_chars = 0
        self._audio_bytes = 0
        self._audio_chunks = 0
        self._close_code = 0
        self._close_reason = ""
        self._buf_meter = StreamMeter("speech", "qwen.buffer", kind="in")
        self._frame_meter = StreamMeter("speech", "qwen.video_tx", kind="in")
        self._delta_meter = StreamMeter("speech", "qwen.sse", kind="out")
        self._rx_types = TypeCounter("speech", "qwen.rx_stats")
        self._drop_meter = StreamMeter("speech", "stream.drop_not_ready", kind="in")
        self._text_rl = RateLimiter(1.0)
        self._unknown_rl = RateLimiter(2.0)
        self._not_ready_since = 0.0
        self._not_ready_rl = RateLimiter(5.0)
        self._last_probe: dict[str, Any] = {}
        self._ws_attempts = 0

    @property
    def generating(self) -> bool:
        return self._busy

    @property
    def ready(self) -> bool:
        return self._ready.is_set() and self._ws is not None

    def stream_snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        not_ready_ms = _age_ms(self._not_ready_since, now) if self._not_ready_since else 0
        snap = {
            "ready": self.ready,
            "speaking": self._busy,
            "out_q": self._out.qsize(),
            "utt_ms": int(1000.0 * (self._utt_bytes / 2.0) / MODEL_RATE) if self._utt_bytes else 0,
            "frame_n": self._frame_n,
            "audio_n": self._audio_n,
            "mic_age_ms": _age_ms(self._t_last_mic, now),
            "frame_age_ms": _age_ms(self._t_last_frame, now),
            "delta_age_ms": _age_ms(self._t_last_delta, now),
            "rx_age_ms": _age_ms(self._t_last_rx, now),
            "ws_attempts": self._ws_attempts,
            "not_ready_ms": not_ready_ms,
        }
        probe = self._last_probe
        if probe:
            snap.update(
                {
                    "model_tcp_ok": bool(probe.get("tcp_ok")),
                    "model_http_ok": bool(probe.get("http_ok")),
                    "model_models_ok": bool(probe.get("models_ok")),
                    "model_probe_detail": str(probe.get("detail") or "")[:120],
                }
            )
        return snap

    async def _send(self, body: dict[str, Any]) -> None:
        if self._ws is None:
            return
        await self._ws.send(json.dumps(body))

    async def _connect_ws(self, url: str, attempts: int = 181) -> int:
        delay = 2.0
        last_err = "unknown"
        self._ws_attempts = 0
        for attempt in range(1, attempts + 1):
            self._ws_attempts = attempt
            try:
                self._ws = await websockets.connect(url, max_size=16_000_000, ping_interval=20)
                if attempt > 1:
                    trace_stream(
                        "speech",
                        "qwen.ws_connected",
                        attempt=attempt,
                        url=url,
                        model=self.model,
                    )
                return attempt
            except Exception as exc:
                last_err = str(exc)[:200]
                probe: dict[str, Any] = {}
                if attempt == 1 or attempt in (3, 5, 10, 20) or attempt % 15 == 0:
                    probe = await probe_model_server()
                    self._last_probe = probe
                    trace_stream(
                        "speech",
                        "qwen.model_probe",
                        attempt=attempt,
                        url=url,
                        error=last_err,
                        **probe,
                    )
                    if not probe.get("tcp_ok"):
                        LOG.warning(
                            "model-server not listening on %s:%s (attempt %s): %s",
                            probe.get("host"),
                            probe.get("port"),
                            attempt,
                            probe.get("detail") or last_err,
                        )
                trace_stream(
                    "speech",
                    "qwen.ws_retry",
                    attempt=attempt,
                    error=last_err,
                    url=url,
                    model_base_url=_model_http_url(),
                    model_tcp_ok=bool(probe.get("tcp_ok")) if probe else None,
                )
                await asyncio.sleep(min(delay, 15.0))
                delay = min(delay * 1.25, 15.0)
        probe = await probe_model_server()
        self._last_probe = probe
        trace_stream(
            "speech",
            "error.qwen_ws_connect_failed",
            attempts=attempts,
            error=last_err,
            url=url,
            **probe,
        )
        LOG.error(
            "model-server video stream never accepted %s after %s attempts: %s; probe=%s",
            url,
            attempts,
            last_err,
            probe.get("detail") or probe,
        )
        raise RuntimeError(f"model-server video stream never accepted {url}: {last_err}")

    async def _send_session_config(self) -> None:
        config: dict[str, Any] = {
            "type": "session.config",
            "modalities": ["text", "audio"],
            "num_frames": self._num_frames,
            "max_frames": self._max_frames,
            "use_audio_in_video": False,
            "enable_frame_filter": True,
            "system_prompt": self.system_prompt,
        }
        await self._send(config)

    async def start(self) -> None:
        url = _model_ws_url()
        trace_stream(
            "speech",
            "qwen.ws_open",
            url=url,
            model=self.model,
            num_frames=self._num_frames,
            max_frames=self._max_frames,
        )
        connected_at = await self._connect_ws(url)
        await self._send_session_config()
        self._t_ready = time.monotonic()
        self._not_ready_since = 0.0
        self._ready.set()
        self._reader = asyncio.create_task(self._read())
        self._hb = asyncio.create_task(self._heartbeat())
        trace_stream(
            "speech",
            "qwen.ready",
            url=url,
            attempt=connected_at,
            speaker=self.speaker,
            model=self.model,
            num_frames=self._num_frames,
            max_frames=self._max_frames,
        )
        trace_stream("speech", "voice.ref_loaded", source=f"qwen_speaker:{self.speaker}")

    async def _reset_socket(self, reason: str) -> None:
        """Drop a hung /v1/video/chat/stream session (empty codec wait-gate)."""
        if self._resetting:
            return
        self._resetting = True
        self._ready.clear()
        self._busy = False
        self._t_query = 0.0
        reader = self._reader
        self._reader = None
        if reader is not None:
            reader.cancel()
            try:
                await reader
            except (asyncio.CancelledError, Exception):
                pass
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        url = _model_ws_url()
        trace_stream("speech", "qwen.reconnect", reason=reason, url=url)
        try:
            await self._connect_ws(url, attempts=12)
            await self._send_session_config()
            self._frame_n = 0
            self._audio_n = 0
            self._utt_bytes = 0
            self._t_ready = time.monotonic()
            self._ready.set()
            self._reader = asyncio.create_task(self._read())
            trace_stream(
                "speech",
                "qwen.ready",
                url=url,
                reconnect=True,
                reason=reason,
                speaker=self.speaker,
                model=self.model,
            )
        except Exception as exc:
            trace_exc("speech", "error.qwen_reconnect", exc, reason=reason)
        finally:
            self._resetting = False

    async def _heartbeat(self) -> None:
        ticks = 0
        try:
            while True:
                await asyncio.sleep(2.0)
                ticks += 1
                if self._ws is None or self._resetting:
                    continue
                try:
                    now = time.monotonic()
                    if (
                        self._busy
                        and self._t_query
                        and not self._first_audio
                        and (now - self._t_query) >= self._stall_s
                    ):
                        age_ms = _age_ms(self._t_query, now)
                        trace_stream(
                            "speech",
                            "qwen.stall",
                            latency_ms=age_ms,
                            text_chars=self._text_chars,
                            frames=self._frame_n,
                        )
                        self._finish_turn("stall_no_audio")
                        await self._reset_socket(f"stall_{age_ms}ms")
                        continue
                    if ticks % 10 == 0:
                        await self._send({"type": "ping"})
                        trace_stream(
                            "speech",
                            "qwen.heartbeat",
                            rx_age_ms=_age_ms(self._t_last_rx, now),
                            delta_age_ms=_age_ms(self._t_last_delta, now),
                            busy=self._busy,
                            frame_n=self._frame_n,
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
                    continue
                try:
                    ev = json.loads(raw)
                except json.JSONDecodeError:
                    trace_stream("speech", "qwen.bad_json", bytes=len(raw) if isinstance(raw, str) else 0)
                    continue
                if isinstance(ev, dict):
                    await self._handle(ev)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            code = getattr(exc, "code", None)
            reason = str(getattr(exc, "reason", "") or exc)[:200]
            self._close_code = int(code) if isinstance(code, int) else 0
            self._close_reason = reason
            self._ready.clear()
            self._busy = False
            trace_exc(
                "speech",
                "error.qwen_ws",
                exc,
                code=self._close_code,
                reason=reason,
                busy=self._busy,
            )

    async def _handle(self, ev: dict[str, Any]) -> None:
        typ = str(ev.get("type") or "")
        self._rx_types.add(typ)
        if typ == "pong" or typ == "video.frame.ack":
            return
        if typ == "response.start":
            if self._drop_output:
                return
            self._busy = True
            self._first_audio = False
            if not self._t_query:
                self._t_query = time.monotonic()
            trace_stream("speech", "qwen.response_start")
            return
        if typ in _AUDIO_DELTA_TYPES:
            await self._on_audio_delta(ev)
            return
        if typ in _TEXT_DELTA_TYPES:
            text = str(ev.get("delta") or ev.get("text") or "")
            if text:
                self._text_n += 1
                self._text_chars += len(text)
                if self._text_n == 1 or self._text_rl.due():
                    trace_stream("speech", "qwen.text_delta", n=self._text_chars, text=text[:80])
            return
        if typ == "response.text.done":
            text = str(ev.get("text") or "")
            if text and (self._text_n <= 1 or self._text_rl.due()):
                trace_stream("speech", "qwen.text_done", chars=len(text), text=text[:80])
            return
        if typ == "video.frames.consumed":
            ids = ev.get("frame_ids") or []
            n = ev.get("frames")
            if n is None:
                n = len(ids) if isinstance(ids, list) else 0
            trace_stream(
                "speech",
                "qwen.frames_consumed",
                frames=n,
                latest_pts_ms=ev.get("latest_pts_ms"),
            )
            return
        if typ in _DONE_TYPES:
            self._finish_turn(typ)
            return
        if typ in ("error", "response.error"):
            msg = str(ev.get("message") or ev.get("error") or ev)[:400]
            trace_stream("speech", "error.qwen", error=msg, type=typ)
            self._finish_turn("error")
            return
        if typ == "session.done":
            self._ready.clear()
            self._busy = False
            trace_stream("speech", "qwen.session_done")
            return
        if self._unknown_rl.due() or not typ:
            keys = ",".join(list(ev.keys())[:12])
            trace_stream("speech", "qwen.event", type=typ or "empty", keys=keys)

    async def _on_audio_delta(self, ev: dict[str, Any]) -> None:
        if self._drop_output:
            return
        pcm, rate = _pcm_from_delta(ev)
        if not pcm:
            return
        now = time.monotonic()
        self._busy = True
        self._audio_bytes += len(pcm)
        self._audio_chunks += 1
        self._t_last_delta = now
        self._delta_meter.add(lines=1, audio_chunks=1, bytes=len(pcm))
        if not self._first_audio:
            self._first_audio = True
            trace_stream(
                "speech",
                "qwen.first_audio",
                latency_ms=_age_ms(self._t_query, now),
                bytes=len(pcm),
                rate=rate,
                frames=self._frame_n,
            )
        await self._out.put((pcm, rate))

    def _finish_turn(self, reason: str) -> None:
        if not self._busy and reason != "error":
            return
        self._busy = False
        latency_ms = _age_ms(self._t_query, time.monotonic())
        self._delta_meter.flush()
        self._rx_types.flush()
        trace_stream(
            "speech",
            "qwen.done",
            reason=reason,
            latency_ms=latency_ms,
            audio_bytes=self._audio_bytes,
            audio_chunks=self._audio_chunks,
            text_chars=self._text_chars,
        )
        self._t_query = 0.0
        self._audio_bytes = 0
        self._audio_chunks = 0
        self._text_n = 0
        self._text_chars = 0
        self._first_audio = False

    async def push_audio(self, pcm16: bytes, sample_rate: int, is_speech: bool = True) -> None:
        if not pcm16:
            return
        if sample_rate != MODEL_RATE:
            if self._in_rs is None:
                self._in_rs = LinearResampler(MODEL_RATE)
            pcm16 = self._in_rs.feed(pcm16, sample_rate)
            if not pcm16:
                return
        self._t_last_mic = time.monotonic()
        if not self.ready:
            now = time.monotonic()
            if not self._not_ready_since:
                self._not_ready_since = now
            self._drop_meter.add(bytes=len(pcm16), frames=1)
            if self._not_ready_rl.due():
                not_ready_ms = _age_ms(self._not_ready_since, now)
                probe = await probe_model_server()
                self._last_probe = probe
                trace_stream(
                    "speech",
                    "qwen.engine_not_ready",
                    not_ready_ms=not_ready_ms,
                    ws_attempts=self._ws_attempts,
                    **probe,
                )
            return
        self._not_ready_since = 0.0
        if is_speech:
            self._utt_bytes += len(pcm16)
        self._audio_n += 1
        self._buf_meter.add(
            bytes=len(pcm16),
            utt_ms=int(1000.0 * (self._utt_bytes / 2.0) / MODEL_RATE),
            is_speech=int(bool(is_speech)),
        )
        await self._send(
            {
                "type": "audio.chunk",
                "data": base64.b64encode(pcm16).decode("ascii"),
            }
        )

    async def push_image(self, jpeg: bytes) -> None:
        if not jpeg:
            return
        if not self.ready:
            now = time.monotonic()
            if not self._not_ready_since:
                self._not_ready_since = now
            self._drop_meter.add(dropped=1, reason="not_ready")
            if self._not_ready_rl.due():
                not_ready_ms = _age_ms(self._not_ready_since, now)
                trace_stream(
                    "speech",
                    "qwen.engine_not_ready",
                    not_ready_ms=not_ready_ms,
                    ws_attempts=self._ws_attempts,
                    dropped="video",
                    **self._last_probe,
                )
            return
        self._not_ready_since = 0.0
        self._frame_n += 1
        self._t_last_frame = time.monotonic()
        pts_ms = int(self._t_last_frame * 1000)
        self._frame_meter.add(frames=1, bytes=len(jpeg))
        await self._send(
            {
                "type": "video.frame",
                "data": base64.b64encode(jpeg).decode("ascii"),
                "frame_id": str(self._frame_n),
                "pts_ms": pts_ms,
            }
        )
        if self._frame_n == 1 or self._frame_n % 25 == 0:
            trace_stream(
                "speech",
                "qwen.video_frame",
                n=self._frame_n,
                bytes=len(jpeg),
                pts_ms=pts_ms,
            )

    async def generate(self) -> None:
        if self._busy or not self.ready:
            return
        utt_ms = int(1000.0 * (self._utt_bytes / 2.0) / MODEL_RATE)
        if self._utt_bytes < int(MODEL_RATE * 2 * _MIN_UTT_S):
            trace_stream("speech", "qwen.request", skipped="too_short", bytes=self._utt_bytes, utt_ms=utt_ms)
            return
        if self._frame_n <= 0:
            trace_stream("speech", "qwen.request", skipped="no_frames", utt_ms=utt_ms)
            return
        self._utt_bytes = 0
        self._drop_output = False
        self._busy = True
        self._first_audio = False
        self._t_query = time.monotonic()
        self._audio_bytes = 0
        self._audio_chunks = 0
        self._text_n = 0
        self._text_chars = 0
        trace_stream(
            "speech",
            "qwen.request",
            utt_ms=utt_ms,
            frames=self._frame_n,
            num_frames=self._num_frames,
            speaker=self.speaker,
        )
        try:
            await self._send({"type": "video.query", "text": _QUERY_TEXT})
        except Exception as exc:
            self._busy = False
            self._t_query = 0.0
            trace_exc("speech", "error.qwen_query", exc)

    async def pull_audio(self) -> list[tuple[bytes, int]]:
        out: list[tuple[bytes, int]] = []
        while True:
            try:
                out.append(self._out.get_nowait())
            except asyncio.QueueEmpty:
                break
        return out

    async def interrupt(self) -> None:
        q = self._out.qsize()
        self._drop_output = True
        self._busy = False
        self._t_query = 0.0
        while not self._out.empty():
            try:
                self._out.get_nowait()
            except asyncio.QueueEmpty:
                break
        trace_stream("speech", "qwen.cancel", reason="interrupt", dropped_q=q)

    async def close(self) -> None:
        await self.interrupt()
        for meter in (self._buf_meter, self._frame_meter, self._delta_meter, self._rx_types, self._drop_meter):
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
                await self._send({"type": "video.done"})
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
            self._ready.clear()
            trace_stream(
                "speech",
                "qwen.ws_close",
                code=self._close_code,
                reason=self._close_reason,
                frame_n=self._frame_n,
                audio_n=self._audio_n,
            )
