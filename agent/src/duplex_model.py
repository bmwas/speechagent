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
from trace import trace, trace_dbg, trace_exc

LOG = logging.getLogger("duplex")
CHUNK_MS = int(os.environ.get("AUDIO_CHUNK_MS", "100"))


def _model_ws_url() -> str:
    base = (os.environ.get("MODEL_BASE_URL") or "http://127.0.0.1:8099").rstrip("/")
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


class MiniCPMDuplexEngine:
    """Stream 16 kHz PCM in, receive 24 kHz PCM out. Model decides listen/speak."""

    input_rate = MODEL_RATE

    def __init__(self, system_prompt: str = "") -> None:
        self.system_prompt = system_prompt.strip()
        self._ws: Any = None
        self._out: asyncio.Queue[tuple[bytes, int]] = asyncio.Queue()
        self._reader: asyncio.Task | None = None
        self._buf = bytearray()
        self._in_resampler = None
        self._speaking = False
        self._last_append = 0.0
        self._append_bytes = 0
        self._append_n = 0
        self._video_pending: str | None = None
        self._played_ms = 0
        self._response_id = ""
        self._t_speak = 0.0
        self._out_rate = 24000

    @property
    def generating(self) -> bool:
        return self._speaking

    async def start(self) -> None:
        url = _model_ws_url()
        model = os.environ.get("MODEL_NAME", "openbmb/MiniCPM-o-4_5")
        ref = _ref_audio_data_url()
        trace("speech", "duplex.ws_open", url=url, model=model, has_ref=bool(ref))
        delay = 2.0
        last_err = "unknown"
        for attempt in range(1, 181):
            try:
                self._ws = await websockets.connect(url, max_size=16_000_000, ping_interval=20)
                break
            except Exception as exc:
                last_err = str(exc)[:200]
                trace("speech", "duplex.ws_retry", attempt=attempt, error=last_err)
                await asyncio.sleep(min(delay, 15.0))
                delay = min(delay * 1.25, 15.0)
        else:
            raise RuntimeError(f"model-server duplex never accepted {url}: {last_err}")
        session: dict[str, Any] = {
            "modalities": ["audio", "text"],
            "instructions": self.system_prompt
            or "You are a helpful, friendly AI assistant. Respond naturally and conversationally.",
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_sample_rate": MODEL_RATE,
            "turn_detection": None,
        }
        extra: dict[str, Any] = {
            "force_listen_count": 0,
            "overlap_policy": "listen_only",
            "playback_commit_policy": "ack_only",
        }
        if ref:
            extra["ref_audio"] = ref
            trace("speech", "voice.ref_loaded", source="REF_AUDIO_PATH")
        else:
            trace("speech", "voice.ref_loaded", source="HT_ref_audio.wav")
        payload = {
            "type": "session.update",
            "session": session,
            "extra_body": extra,
            "model": model,
        }
        await self._ws.send(json.dumps(payload))
        self._reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    if raw:
                        await self._out.put((raw, self._out_rate))
                    continue
                try:
                    ev = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                await self._handle(ev)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            trace_exc("speech", "error.duplex_ws", exc)

    async def _handle(self, ev: dict[str, Any]) -> None:
        typ = str(ev.get("type") or "")
        if typ == "session.created":
            trace("speech", "duplex.session_created", session_id=str(ev.get("session", {}).get("id", ""))[:16])
            return
        if typ in ("response.created",):
            self._response_id = str(ev.get("response", {}).get("id") or ev.get("response_id") or "")
            self._t_speak = time.monotonic()
            return
        if typ in ("response.audio.delta", "response.output_audio.delta", "audio.delta"):
            b64 = ev.get("delta") or ev.get("audio") or ""
            if not b64:
                return
            try:
                pcm = base64.b64decode(b64)
            except Exception:
                return
            if pcm[:4] == b"RIFF":
                pcm = pcm[44:]
            if not self._speaking:
                self._speaking = True
                lat = int((time.monotonic() - self._t_speak) * 1000) if self._t_speak else -1
                trace("speech", "duplex.speak", latency_ms=lat, bytes=len(pcm))
            await self._out.put((pcm, self._out_rate))
            return
        if typ in ("response.listen",) or ev.get("decision") == "listen":
            trace("speech", "duplex.listen")
            self._speaking = False
            return
        if typ in ("response.done", "response.completed"):
            status = str((ev.get("response") or {}).get("status") or "")
            trace("speech", "duplex.audio_delta", done=True, status=status, played_ms=self._played_ms)
            self._speaking = False
            return
        if typ in ("error", "response.error"):
            trace("speech", "error.duplex", error=str(ev.get("error") or ev)[:300])

    async def push_audio(self, pcm16: bytes, sample_rate: int) -> None:
        from audio import LinearResampler

        if self._ws is None or not pcm16:
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
            await self._append(piece)

    async def _append(self, pcm16: bytes) -> None:
        if self._ws is None:
            return
        body: dict[str, Any] = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm16).decode("ascii"),
            "sample_rate": MODEL_RATE,
        }
        if self._video_pending:
            body["video_frames"] = [self._video_pending]
            self._video_pending = None
        await self._ws.send(json.dumps(body))
        self._append_n += 1
        self._append_bytes += len(pcm16)
        now = time.monotonic()
        if now - self._last_append >= 1.0:
            trace(
                "speech",
                "duplex.append",
                chunks=self._append_n,
                bytes=self._append_bytes,
                rms=pcm_rms(pcm16),
                video_attached="video_frames" in body,
            )
            self._append_n = 0
            self._append_bytes = 0
            self._last_append = now
        else:
            trace_dbg("speech", "audio.chunk", bytes=len(pcm16), rms=pcm_rms(pcm16))

    async def push_image(self, jpeg: bytes) -> None:
        if not jpeg:
            return
        if self._speaking:
            trace_dbg("speech", "video.chunk", bytes=len(jpeg), suppressed=True)
            return
        b64 = base64.b64encode(jpeg).decode("ascii")
        self._video_pending = f"data:image/jpeg;base64,{b64}"

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
                await self._ws.send(
                    json.dumps(
                        {
                            "type": "playback.ack",
                            "played_ms": self._played_ms,
                            "response_id": self._response_id,
                        }
                    )
                )
            except Exception:
                pass
        return out

    async def interrupt(self) -> None:
        self._speaking = False
        while not self._out.empty():
            try:
                self._out.get_nowait()
            except asyncio.QueueEmpty:
                break
        if self._ws is None:
            return
        try:
            await self._ws.send(json.dumps({"type": "response.cancel"}))
            trace("speech", "duplex.cancel")
        except Exception as exc:
            trace_exc("speech", "error.duplex_cancel", exc)

    async def close(self) -> None:
        if self._reader:
            self._reader.cancel()
            try:
                await self._reader
            except (asyncio.CancelledError, Exception):
                pass
            self._reader = None
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps({"type": "session.close"}))
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
            trace("speech", "duplex.ws_close")
