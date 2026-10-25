"""MiniCPM-o 4.5 native duplex client over OpenBMB ``/v1/worker/duplex``.

Protocol (worker.py ``_handle_remote_backend_runtime_ws``):

    -> session.init { payload: { system_prompt, voice, ref_audio_base64, config } }
    <- session.created
    -> input.append { input: { audio, video_frames?, force_listen } }   # 1 s float32
    <- response.output.delta { kind: listen | text | audio }
    -> session.close
    <- session.closed

Audio in: 16 kHz mono little-endian float32 PCM, base64.
Audio out: 24 kHz mono little-endian float32 PCM, base64.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import struct
import time
import wave
from pathlib import Path
from typing import Any

import websockets

from audio import MODEL_RATE, pcm_rms
from duplex_model import _find_bundled_ref_audio
from trace import StreamMeter, TypeCounter, hires_enabled, trace_exc, trace_stream
from voices import DEFAULT_VOICE_ID

LOG = logging.getLogger("native")
OUT_RATE = 24000
CHUNK_MS = int(os.environ.get("NATIVE_CHUNK_MS", "1000"))
SESSION_CAP_S = float(os.environ.get("NATIVE_SESSION_S", "300"))
TTS_REF_MAX_S = float(os.environ.get("NATIVE_TTS_REF_MAX_S", "10"))
TTS_REF_MIN_S = float(os.environ.get("NATIVE_TTS_REF_MIN_S", "5"))
# Consecutive listen units while the user is still talking → session is stuck.
LISTEN_LOCK_N = int(os.environ.get("NATIVE_LISTEN_LOCK_N", "5"))
LISTEN_LOCK_RMS = float(os.environ.get("NATIVE_LISTEN_LOCK_RMS", "400"))
# OpenBMB default is 1.0. Values <1 make the model eager to speak and, because
# streaming_generate refuses <|listen|> until current_turn_ended, it will
# ramble until the client sends force_listen.
LISTEN_PROB_SCALE = float(os.environ.get("NATIVE_LISTEN_PROB_SCALE", "1.0"))
# Consecutive 1 s speak units with no listen → force_listen to close the turn.
SPEAK_YIELD_N = int(os.environ.get("NATIVE_SPEAK_YIELD_N", "10"))
# Consecutive 1 s mic units at user RMS while the bot is speaking → barge
# without waiting for Silero (echo hold + bot audio often suppress START).
SPEAK_BARGE_N = int(os.environ.get("NATIVE_SPEAK_BARGE_N", "2"))
SPEAK_BARGE_RMS = float(os.environ.get("NATIVE_SPEAK_BARGE_RMS", "400"))


def _worker_ws_url() -> str:
    # Native worker is :22400. MODEL_BASE_URL often still points at vLLM :8199
    # after a backend switch; remap that default rather than talking to the
    # wrong process. NATIVE_WORKER_URL wins when set.
    base = (os.environ.get("NATIVE_WORKER_URL") or os.environ.get("MODEL_BASE_URL") or "").strip().rstrip("/")
    if not base or ":8199" in base:
        base = "http://127.0.0.1:22400"
    if base.startswith("http://"):
        base = "ws://" + base[len("http://") :]
    elif base.startswith("https://"):
        base = "wss://" + base[len("https://") :]
    elif not base.startswith("ws"):
        base = "ws://" + base.lstrip("/")
    if base.endswith("/v1/worker/duplex"):
        return base
    return f"{base}/v1/worker/duplex"


def _pcm16_to_f32le(pcm16: bytes) -> bytes:
    n = len(pcm16) // 2
    if n <= 0:
        return b""
    samples = struct.unpack("<" + "h" * n, pcm16[: n * 2])
    return struct.pack("<" + "f" * n, *[s / 32768.0 for s in samples])


def _f32le_to_pcm16(raw: bytes) -> bytes:
    n = len(raw) // 4
    if n <= 0:
        return b""
    floats = struct.unpack("<" + "f" * n, raw[: n * 4])
    out = []
    for x in floats:
        if x > 1.0:
            x = 1.0
        elif x < -1.0:
            x = -1.0
        out.append(int(x * 32767.0))
    return struct.pack("<" + "h" * n, *out)


def _age_ms(stamp: float, now: float) -> int:
    if stamp <= 0:
        return -1
    return int((now - stamp) * 1000)


def _load_ref_f32le(path: Path | None) -> tuple[str, str, str, dict[str, Any]]:
    """Return (llm_b64, tts_b64, source, meta). TTS clip is capped at TTS_REF_MAX_S."""
    if path is not None:
        if not path.is_file():
            raise FileNotFoundError(f"ref_audio missing: {path}")
        source = str(path)
    else:
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
                    "Native MiniCPM duplex requires ref_audio. Set REF_AUDIO_PATH "
                    "to a 16 kHz mono WAV, or keep assets/HT_ref_audio.wav in the "
                    "HuggingFace snapshot under /models/hf."
                )
            source = str(path)
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        rate = handle.getframerate()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError(f"ref_audio must be 16-bit PCM, got sampwidth={width}: {path}")
    n = len(frames) // 2
    samples = list(struct.unpack("<" + "h" * n, frames[: n * 2]))
    if channels > 1:
        mono: list[int] = []
        for i in range(0, len(samples) - channels + 1, channels):
            mono.append(int(sum(samples[i : i + channels]) / channels))
        samples = mono
    if rate != MODEL_RATE and rate > 0:
        # Linear resample to 16 kHz. Catalog voices are already 16 kHz.
        ratio = rate / float(MODEL_RATE)
        new_n = max(1, int(round(len(samples) / ratio)))
        resampled: list[int] = []
        pos = 0.0
        last = len(samples) - 1
        for _ in range(new_n):
            i = int(pos)
            if i >= last:
                resampled.append(samples[-1])
                break
            frac = pos - i
            resampled.append(int(samples[i] + (samples[i + 1] - samples[i]) * frac))
            pos += ratio
        samples = resampled
    duration_s = len(samples) / float(MODEL_RATE)
    llm_raw = struct.pack("<" + "f" * len(samples), *[s / 32768.0 for s in samples])
    tts_n = min(len(samples), int(TTS_REF_MAX_S * MODEL_RATE))
    tts_samples = samples[:tts_n]
    tts_raw = struct.pack("<" + "f" * len(tts_samples), *[s / 32768.0 for s in tts_samples])
    meta = {
        "bytes": path.stat().st_size,
        "duration_ms": int(duration_s * 1000),
        "tts_duration_ms": int(len(tts_samples) / float(MODEL_RATE) * 1000),
        "rate": MODEL_RATE,
        "sha256": "",
    }
    if duration_s + 1e-6 < TTS_REF_MIN_S:
        LOG.info("ref_audio is %.2fs (< %.1fs recommended TTS floor); sending as-is", duration_s, TTS_REF_MIN_S)
    return (
        base64.b64encode(llm_raw).decode("ascii"),
        base64.b64encode(tts_raw).decode("ascii"),
        source,
        meta,
    )


class NativeMiniCPMEngine:
    """Stream 16 kHz PCM in (1 s units), receive 24 kHz PCM out."""

    input_rate = MODEL_RATE

    def __init__(
        self,
        system_prompt: str = "",
        voice_id: str = DEFAULT_VOICE_ID,
        ref_path: Path | str | None = None,
    ) -> None:
        self.system_prompt = system_prompt.strip()
        self.voice_id = (voice_id or DEFAULT_VOICE_ID).strip().lower() or DEFAULT_VOICE_ID
        self._ref_path = Path(ref_path) if ref_path else None
        self._ws: Any = None
        self._out: asyncio.Queue[tuple[bytes, int]] = asyncio.Queue()
        self._reader: asyncio.Task | None = None
        self._watchdog: asyncio.Task | None = None
        self._buf = bytearray()
        self._speaking = False
        self._ready = asyncio.Event()
        self._handshake = asyncio.Event()
        self._handshake_error = ""
        self._closing = False
        self._force_listen = False
        self._video_pending: str | None = None
        self._video_n = 0
        self._append_n = 0
        self._delta_n = 0
        self._delta_bytes = 0
        self._t_ready = 0.0
        self._t_session = 0.0
        self._t_last_mic = 0.0
        self._t_last_append = 0.0
        self._t_last_delta = 0.0
        self._t_last_rx = 0.0
        self._t_speak = 0.0
        self._first_audio = False
        self._last_append_rms = 0
        self._listen_speech_streak = 0
        self._speak_streak = 0
        self._speak_user_streak = 0
        self._session_id = ""
        self._reconnect_reason = ""
        self._last_reconnect_reason = ""
        self._last_reconnect = 0.0
        self._session_generation = 0
        self._unit_cost_ms: list[float] = []
        self._rx_types = TypeCounter("speech", "native.rx_stats")
        self._append_meter = StreamMeter("speech", "native.append", kind="in")
        self._delta_meter = StreamMeter("speech", "native.audio_delta", kind="out")
        self._drop_meter = StreamMeter("speech", "stream.drop_not_ready", reason="ws_not_ready")
        self._chunk_samples = int(MODEL_RATE * CHUNK_MS / 1000)
        self._chunk_bytes = self._chunk_samples * 2

    @property
    def generating(self) -> bool:
        return self._speaking

    @property
    def ready(self) -> bool:
        return self._ws is not None and self._ready.is_set()

    def stream_snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        costs = self._unit_cost_ms[-32:]
        median = sorted(costs)[len(costs) // 2] if costs else -1.0
        return {
            "ws_open": self._ws is not None,
            "ready": self._ready.is_set(),
            "speaking": self._speaking,
            "path": "native",
            "session_id": self._session_id[:24],
            "out_q": self._out.qsize(),
            "buf_ms": int(1000.0 * (len(self._buf) / 2.0) / MODEL_RATE) if self._buf else 0,
            "delta_n": self._delta_n,
            "append_n": self._append_n,
            "mic_age_ms": _age_ms(self._t_last_mic, now),
            "append_age_ms": _age_ms(self._t_last_append, now),
            "delta_age_ms": _age_ms(self._t_last_delta, now),
            "rx_age_ms": _age_ms(self._t_last_rx, now),
            "video_n": self._video_n,
            "video_pending": bool(self._video_pending),
            "session_age_s": round(now - self._t_session, 1) if self._t_session else -1,
            "session_generation": self._session_generation,
            "reconnect_reason": self._last_reconnect_reason[:80],
            "force_listen": int(self._force_listen),
            "listen_speech_streak": self._listen_speech_streak,
            "speak_streak": self._speak_streak,
            "speak_user_streak": self._speak_user_streak,
            "cost_all_ms_p50": round(median, 1) if median >= 0 else -1,
            "voice_id": self.voice_id,
        }

    def _handshake_ref_path(self) -> Path | None:
        if self.voice_id == DEFAULT_VOICE_ID:
            return None
        return self._ref_path

    async def _send(self, body: dict[str, Any]) -> None:
        if self._ws is None:
            return
        await self._ws.send(json.dumps(body))

    async def start(self) -> None:
        self._closing = False
        await self._connect_and_handshake()
        if self._watchdog is None or self._watchdog.done():
            self._watchdog = asyncio.create_task(self._watchdog_loop())

    async def _connect_and_handshake(self) -> None:
        url = _worker_ws_url()
        llm_b64, tts_b64, ref_source, ref_meta = _load_ref_f32le(self._handshake_ref_path())
        prompt = self.system_prompt or (
            "Clone the voice in the provided audio prompt. Please assist users "
            "while maintaining this voice style. Always reply in English."
        )
        trace_stream("speech", "native.ws_open", url=url, has_ref=True, ref_source=ref_source[-80:])
        delay = 2.0
        last_err = "unknown"
        connected_at = 0
        for attempt in range(1, 181):
            if self._closing:
                raise RuntimeError("native duplex closed before connect")
            try:
                self._ws = await websockets.connect(url, max_size=128 * 1024 * 1024, ping_interval=20)
                connected_at = attempt
                break
            except Exception as exc:
                last_err = str(exc)[:200]
                trace_stream("speech", "native.ws_retry", attempt=attempt, error=last_err, url=url)
                await asyncio.sleep(min(delay, 15.0))
                delay = min(delay * 1.25, 15.0)
        else:
            raise RuntimeError(f"native worker never accepted {url}: {last_err}")

        trace_stream("speech", "native.ws_connected", url=url, attempt=connected_at)
        payload: dict[str, Any] = {
            "mode": "full_duplex",
            "system_prompt": prompt,
            "instructions": prompt,
            "config": {
                "generate_audio": True,
                "chunk_ms": CHUNK_MS,
                "sample_rate": MODEL_RATE,
                "temperature": 0.7,
                "force_listen_count": 3,
                "listen_prob_scale": LISTEN_PROB_SCALE,
            },
            "voice": {
                "ref_audio_base64": llm_b64,
                "tts_ref_audio_base64": tts_b64,
            },
            # Legacy field names (pre-voice object). Harmless if ignored.
            "ref_audio_base64": llm_b64,
            "tts_ref_audio_base64": tts_b64,
        }
        trace_stream(
            "speech",
            "voice.ref_loaded",
            voice_id=self.voice_id,
            source=ref_source[-120:],
            bytes=int(ref_meta.get("bytes") or 0),
            duration_ms=int(ref_meta.get("duration_ms") or -1),
            tts_duration_ms=int(ref_meta.get("tts_duration_ms") or -1),
            backend="native",
        )
        self._handshake.clear()
        self._handshake_error = ""
        self._ready.clear()
        self._reader = asyncio.create_task(self._read())
        await self._send({"type": "session.init", "payload": payload})
        try:
            await asyncio.wait_for(self._handshake.wait(), timeout=60.0)
        except asyncio.TimeoutError as exc:
            raise RuntimeError("native worker never sent session.created") from exc
        if self._handshake_error:
            raise RuntimeError(f"native handshake failed: {self._handshake_error}")
        if not self._ready.is_set():
            raise RuntimeError("native worker never sent session.created")
        self._session_generation += 1
        self._listen_speech_streak = 0
        self._t_session = time.monotonic()
        self._t_ready = self._t_session
        self._first_audio = False
        trace_stream(
            "speech",
            "native.session_created",
            session_id=self._session_id[:24],
            session_generation=self._session_generation,
            reconnect_reason=self._last_reconnect_reason[:80],
        )

    async def _read(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                self._t_last_rx = time.monotonic()
                try:
                    ev = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(ev, dict):
                    continue
                await self._on_event(ev)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closing:
                trace_exc("speech", "error.native_read", exc)
                self._reconnect_reason = "ws_closed"
        finally:
            self._ready.clear()

    async def _on_event(self, ev: dict[str, Any]) -> None:
        kind = str(ev.get("type") or "")
        self._rx_types.add(kind or "unknown")
        if kind == "session.created":
            self._session_id = str(ev.get("session_id") or "")
            self._ready.set()
            self._handshake.set()
            return
        if kind == "error":
            err = ev.get("error")
            if isinstance(err, dict):
                msg = str(err.get("message") or err.get("code") or err)[:400]
            else:
                msg = str(err or ev)[:400]
            self._handshake_error = msg
            self._handshake.set()
            trace_stream("speech", "native.error", message=msg)
            return
        if kind == "session.closed":
            reason = str(ev.get("reason") or "closed")
            trace_stream("speech", "native.session_closed", reason=reason[:80])
            self._ready.clear()
            if not self._closing:
                self._reconnect_reason = reason or "session_closed"
            return
        if kind != "response.output.delta":
            if hires_enabled():
                trace_stream("speech", "native.ws_rx", type=kind[:80])
            return
        delta_kind = str(ev.get("kind") or "")
        metrics = ev.get("metrics") if isinstance(ev.get("metrics"), dict) else {}
        cost = metrics.get("cost_all_ms")
        if isinstance(cost, (int, float)):
            self._unit_cost_ms.append(float(cost))
            if len(self._unit_cost_ms) > 256:
                del self._unit_cost_ms[:128]
        if delta_kind == "listen":
            was = self._speaking
            self._speaking = False
            self._speak_streak = 0
            self._speak_user_streak = 0
            if self._last_append_rms >= LISTEN_LOCK_RMS:
                self._listen_speech_streak += 1
            else:
                self._listen_speech_streak = 0
            if (
                LISTEN_LOCK_N > 0
                and self._listen_speech_streak >= LISTEN_LOCK_N
                and not self._reconnect_reason
            ):
                self._reconnect_reason = "listen_lock"
            trace_stream(
                "speech",
                "native.listen",
                was_speaking=int(was),
                dropped_q=self._out.qsize(),
                kv=int(metrics.get("kv_cache_length") or 0),
                listen_speech_streak=self._listen_speech_streak,
                append_rms=int(self._last_append_rms),
                cost_all_ms=round(float(cost), 1) if isinstance(cost, (int, float)) else -1,
            )
            return
        if delta_kind == "text":
            text = str(ev.get("text") or "")
            if text:
                self._listen_speech_streak = 0
                if not self._speaking:
                    self._speaking = True
                    self._t_speak = time.monotonic()
                    trace_stream("speech", "native.speak", text=text[:80])
                if hires_enabled():
                    trace_stream("speech", "native.text", chars=len(text), text=text[:80])
            return
        if delta_kind == "audio":
            audio_b64 = ev.get("audio") or ""
            if not audio_b64:
                return
            try:
                raw = base64.b64decode(audio_b64)
            except Exception:
                return
            pcm = _f32le_to_pcm16(raw)
            if not pcm:
                return
            now = time.monotonic()
            if not self._speaking:
                self._speaking = True
                self._t_speak = now
            self._listen_speech_streak = 0
            self._speak_streak += 1
            self._delta_n += 1
            self._delta_bytes += len(pcm)
            self._t_last_delta = now
            dur_ms = int(1000.0 * (len(pcm) / 2.0) / OUT_RATE)
            if not self._first_audio:
                self._first_audio = True
                latency_ms = int((now - self._t_ready) * 1000) if self._t_ready else -1
                trace_stream(
                    "speech",
                    "native.first_audio",
                    latency_ms=latency_ms,
                    bytes=len(pcm),
                    duration_ms=dur_ms,
                )
            self._delta_meter.add(chunks=1, bytes=len(pcm), duration_ms=dur_ms)
            if hires_enabled():
                trace_stream(
                    "speech",
                    "native.audio_delta",
                    bytes=len(pcm),
                    duration_ms=dur_ms,
                    qsize=self._out.qsize(),
                    rms=pcm_rms(pcm),
                    cost_all_ms=round(float(cost), 1) if isinstance(cost, (int, float)) else -1,
                )
            await self._out.put((pcm, OUT_RATE))

    async def _watchdog_loop(self) -> None:
        try:
            while not self._closing:
                await asyncio.sleep(1.0)
                if self._closing:
                    return
                reason = self._reconnect_reason
                now = time.monotonic()
                if not reason and self._t_session and (now - self._t_session) >= SESSION_CAP_S:
                    reason = "session_cap"
                if not reason:
                    continue
                if now - self._last_reconnect < 5.0:
                    continue
                self._reconnect_reason = ""
                self._last_reconnect = now
                try:
                    await self._reconnect(reason)
                except Exception as exc:
                    trace_exc("speech", "error.native_reconnect", exc, reason=reason)
        except asyncio.CancelledError:
            raise

    async def _reconnect(self, reason: str) -> None:
        self._last_reconnect_reason = reason
        trace_stream(
            "speech",
            "native.reconnect",
            reason=reason[:80],
            session_generation=self._session_generation,
        )
        self._ready.clear()
        self._speaking = False
        self._speak_streak = 0
        self._speak_user_streak = 0
        await self._teardown_socket(cancel_watchdog=False)
        self._buf.clear()
        self._force_listen = False
        await self._connect_and_handshake()
        trace_stream(
            "speech",
            "native.reconnected",
            reason=reason[:80],
            session_generation=self._session_generation,
        )

    async def _teardown_socket(self, *, cancel_watchdog: bool) -> None:
        reader = self._reader
        self._reader = None
        if reader is not None and not reader.done():
            reader.cancel()
            try:
                await reader
            except (asyncio.CancelledError, Exception):
                pass
        if cancel_watchdog:
            dog = self._watchdog
            self._watchdog = None
            if dog is not None and not dog.done():
                dog.cancel()
                try:
                    await dog
                except (asyncio.CancelledError, Exception):
                    pass
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                await ws.send(json.dumps({"type": "session.close", "reason": "user_stop"}))
            except Exception:
                pass
            try:
                await ws.close()
            except Exception:
                pass
        trace_stream("speech", "native.ws_close")

    async def push_audio(self, pcm16: bytes, sample_rate: int, is_speech: bool = True) -> None:
        del is_speech  # native duplex has no VAD commit; we stream every 1 s unit
        if not pcm16:
            return
        self._t_last_mic = time.monotonic()
        if not self.ready:
            self._drop_meter.add(dropped=1, bytes=len(pcm16))
            return
        if sample_rate != MODEL_RATE:
            # SpeechSession already resamples to MODEL_RATE.
            return
        self._buf.extend(pcm16)
        while len(self._buf) >= self._chunk_bytes:
            piece = bytes(self._buf[: self._chunk_bytes])
            del self._buf[: self._chunk_bytes]
            await self._append(piece)

    def _drain_out(self) -> int:
        q = self._out.qsize()
        while not self._out.empty():
            try:
                self._out.get_nowait()
            except asyncio.QueueEmpty:
                break
        return q

    def _request_listen(self, reason: str) -> None:
        was = self._speaking
        q = self._drain_out()
        self._speaking = False
        self._force_listen = True
        if reason == "interrupt":
            return
        trace_stream(
            "speech",
            "native.speak_yield",
            reason=reason[:80],
            was_speaking=int(was),
            dropped_q=q,
            speak_streak=self._speak_streak,
            speak_user_streak=self._speak_user_streak,
            append_rms=int(self._last_append_rms),
        )

    async def _append(self, pcm16: bytes) -> None:
        if self._ws is None:
            return
        rms = pcm_rms(pcm16)
        if self._speaking and rms >= SPEAK_BARGE_RMS:
            self._speak_user_streak += 1
            if SPEAK_BARGE_N > 0 and self._speak_user_streak >= SPEAK_BARGE_N:
                self._request_listen("speak_barge")
        else:
            self._speak_user_streak = 0
        if (
            SPEAK_YIELD_N > 0
            and self._speak_streak >= SPEAK_YIELD_N
            and not self._force_listen
        ):
            self._request_listen("speak_yield")
        audio_b64 = base64.b64encode(_pcm16_to_f32le(pcm16)).decode("ascii")
        body: dict[str, Any] = {
            "audio": audio_b64,
            "force_listen": bool(self._force_listen),
        }
        if self._video_pending:
            body["video_frames"] = [self._video_pending]
            jpeg_chars = len(self._video_pending)
            self._video_pending = None
            self._video_n += 1
            trace_stream(
                "speech",
                "native.video_attach",
                n=self._video_n,
                b64_chars=jpeg_chars,
            )
        force = self._force_listen
        self._force_listen = False
        await self._send({"type": "input.append", "input": body})
        now = time.monotonic()
        self._append_n += 1
        self._t_last_append = now
        self._last_append_rms = rms
        self._append_meter.add(chunks=1, bytes=len(pcm16), rms=rms)
        trace_stream(
            "speech",
            "native.append",
            n=self._append_n,
            bytes=len(pcm16),
            duration_ms=CHUNK_MS,
            force_listen=int(force),
            rms=int(rms),
        )

    async def push_image(self, jpeg: bytes) -> None:
        if not jpeg:
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
        return out

    async def interrupt(self) -> None:
        was = self._speaking
        q = self._out.qsize()
        self._speak_streak = 0
        self._speak_user_streak = 0
        self._request_listen("interrupt")
        trace_stream(
            "speech",
            "native.interrupt",
            was_speaking=int(was),
            dropped_q=q,
        )

    async def close(self) -> None:
        self._closing = True
        self._reconnect_reason = ""
        for meter in (self._rx_types, self._append_meter, self._delta_meter, self._drop_meter):
            try:
                meter.flush()
            except Exception:
                pass
        await self._teardown_socket(cancel_watchdog=True)
        self._ready.clear()
