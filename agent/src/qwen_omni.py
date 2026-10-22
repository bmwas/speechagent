"""Qwen2.5-Omni-3B turn-based speech via ``/v1/chat/completions`` streaming."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import time
import wave
from typing import Any

import aiohttp

from audio import MODEL_RATE, LinearResampler
from trace import RateLimiter, StreamMeter, TypeCounter, trace_exc, trace_stream, trace_turn

LOG = logging.getLogger("qwen")


def _base_url() -> str:
    return (os.environ.get("MODEL_BASE_URL") or "http://127.0.0.1:8199").rstrip("/")


def _wav_b64(pcm16: bytes, rate: int) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm16)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_audio_delta(encoded: str) -> tuple[bytes, int]:
    raw = base64.b64decode(encoded)
    if raw[:4] == b"RIFF":
        with wave.open(io.BytesIO(raw), "rb") as w:
            return w.readframes(w.getnframes()), w.getframerate()
    return raw, 24000


def _extract_text(chunk: dict[str, Any]) -> str:
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str) and content and not delta.get("audio"):
            if len(content) < 80 or " " in content:
                return content
        if isinstance(delta.get("text"), str):
            return str(delta.get("text"))
    return ""


class QwenOmniEngine:
    """Accumulate a user utterance, then stream text+audio back."""

    input_rate = MODEL_RATE

    def __init__(self, system_prompt: str = "") -> None:
        self.system_prompt = system_prompt.strip()
        self.speaker = os.environ.get("QWEN_SPEAKER", "Chelsie").strip() or "Chelsie"
        self.model = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-Omni-3B")
        self._session: aiohttp.ClientSession | None = None
        self._out: asyncio.Queue[tuple[bytes, int]] = asyncio.Queue()
        self._utt = bytearray()
        self._jpeg: bytes | None = None
        self._in_rs: LinearResampler | None = None
        self._busy = False
        self._http_ready = False
        self._task: asyncio.Task | None = None
        self._t_last_mic = 0.0
        self._t_last_delta = 0.0
        self._sse_n = 0
        self._buf_meter = StreamMeter("speech", "qwen.buffer", kind="in")
        self._sse_meter = StreamMeter("speech", "qwen.sse", kind="out")
        self._sse_types = TypeCounter("speech", "qwen.rx_stats")
        self._text_rl = RateLimiter(1.0)

    @property
    def generating(self) -> bool:
        return self._busy

    @property
    def ready(self) -> bool:
        return self._http_ready and self._session is not None and not self._session.closed

    def stream_snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        utt_ms = int(1000.0 * (len(self._utt) / 2.0) / MODEL_RATE) if self._utt else 0
        return {
            "ready": self.ready,
            "speaking": self._busy,
            "out_q": self._out.qsize(),
            "utt_ms": utt_ms,
            "has_jpeg": bool(self._jpeg),
            "mic_age_ms": int((now - self._t_last_mic) * 1000) if self._t_last_mic else -1,
            "delta_age_ms": int((now - self._t_last_delta) * 1000) if self._t_last_delta else -1,
            "sse_n": self._sse_n,
        }

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120))
        delay = 2.0
        last_err = "unknown"
        url = f"{_base_url()}/v1/models"
        for attempt in range(1, 181):
            try:
                async with self._session.get(url) as resp:
                    if resp.status < 500:
                        break
                    last_err = f"HTTP {resp.status}"
            except Exception as exc:
                last_err = str(exc)[:200]
            trace_stream("speech", "qwen.ready_retry", attempt=attempt, error=last_err, url=url)
            await asyncio.sleep(min(delay, 15.0))
            delay = min(delay * 1.25, 15.0)
        else:
            raise RuntimeError(f"model-server never ready at {url}: {last_err}")
        self._http_ready = True
        trace_stream("speech", "voice.ref_loaded", source=f"qwen_speaker:{self.speaker}")
        trace_stream("speech", "qwen.ready", url=url, speaker=self.speaker, model=self.model)

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
        self._utt.extend(pcm16)
        self._buf_meter.add(bytes=len(pcm16), utt_ms=int(1000.0 * (len(self._utt) / 2.0) / MODEL_RATE), is_speech=int(bool(is_speech)))

    async def push_image(self, jpeg: bytes) -> None:
        if jpeg:
            self._jpeg = jpeg

    async def generate(self) -> None:
        if self._busy or self._session is None:
            return
        pcm = bytes(self._utt)
        self._utt.clear()
        jpeg = self._jpeg
        if len(pcm) < MODEL_RATE * 2 * 0.2:
            trace_stream("speech", "qwen.request", skipped="too_short", bytes=len(pcm))
            return
        self._busy = True
        trace_stream(
            "speech",
            "qwen.request",
            bytes=len(pcm),
            utt_ms=int(1000.0 * (len(pcm) / 2.0) / MODEL_RATE),
            jpeg=bool(jpeg),
            speaker=self.speaker,
        )
        self._task = asyncio.create_task(self._request(pcm, jpeg))

    async def _request(self, pcm: bytes, jpeg: bytes | None) -> None:
        assert self._session is not None
        content: list[dict[str, Any]] = [
            {
                "type": "input_audio",
                "input_audio": {"data": _wav_b64(pcm, MODEL_RATE), "format": "wav"},
            }
        ]
        if jpeg:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(jpeg).decode('ascii')}"},
                }
            )
        content.append({"type": "text", "text": "Reply in spoken English. Keep it concise."})
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": content})
        body = {
            "model": self.model,
            "messages": messages,
            "modalities": ["text", "audio"],
            "audio": {"voice": self.speaker, "format": "wav"},
            "stream": True,
            "extra_body": {"speaker": self.speaker},
        }
        url = f"{_base_url()}/v1/chat/completions"
        t0 = time.monotonic()
        first = True
        audio_bytes = 0
        audio_chunks = 0
        text_chars = 0
        sse_lines = 0
        last_chunk = t0
        with trace_turn("speech", event="qwen", bytes=len(pcm), jpeg=bool(jpeg), speaker=self.speaker):
            try:
                async with self._session.post(url, json=body) as resp:
                    if resp.status >= 400:
                        text = await resp.text()
                        trace_stream("speech", "error.qwen_http", status=resp.status, body=text[:300])
                        return
                    trace_stream("speech", "qwen.sse_open", status=resp.status, latency_ms=int((time.monotonic() - t0) * 1000))
                    async for raw in resp.content:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            continue
                        sse_lines += 1
                        self._sse_n += 1
                        data = line[5:].strip()
                        if data == "[DONE]":
                            trace_stream("speech", "qwen.sse_done", sse_lines=sse_lines, audio_chunks=audio_chunks)
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        modality = str(chunk.get("modality") or chunk.get("object") or "")
                        if modality:
                            self._sse_types.add(modality)
                        text_piece = _extract_text(chunk)
                        if text_piece:
                            text_chars += len(text_piece)
                            if text_chars == len(text_piece) or self._text_rl.due():
                                trace_stream("speech", "qwen.text_delta", n=text_chars, text=text_piece[:80])
                        pcm_out, rate = self._extract_audio(chunk)
                        now = time.monotonic()
                        stall_ms = int((now - last_chunk) * 1000)
                        last_chunk = now
                        self._sse_meter.add(lines=1, audio_chunks=int(bool(pcm_out)), bytes=len(pcm_out), stall_ms=stall_ms)
                        if not pcm_out:
                            continue
                        audio_bytes += len(pcm_out)
                        audio_chunks += 1
                        self._t_last_delta = now
                        if first:
                            first = False
                            trace_stream(
                                "speech",
                                "qwen.first_audio",
                                latency_ms=int((now - t0) * 1000),
                                bytes=len(pcm_out),
                                rate=rate,
                                sse_lines=sse_lines,
                            )
                        await self._out.put((pcm_out, rate))
            except asyncio.CancelledError:
                trace_stream("speech", "qwen.cancel", latency_ms=int((time.monotonic() - t0) * 1000), audio_bytes=audio_bytes)
                raise
            except Exception as exc:
                trace_exc("speech", "error.qwen_stream", exc)
            finally:
                self._busy = False
                self._sse_meter.flush()
                self._sse_types.flush()
                trace_stream(
                    "speech",
                    "qwen.done",
                    latency_ms=int((time.monotonic() - t0) * 1000),
                    audio_bytes=audio_bytes,
                    audio_chunks=audio_chunks,
                    text_chars=text_chars,
                    sse_lines=sse_lines,
                )

    def _extract_audio(self, chunk: dict[str, Any]) -> tuple[bytes, int]:
        modality = chunk.get("modality") or chunk.get("object")
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content") or delta.get("audio") or ""
            if not content:
                continue
            if modality == "audio" or delta.get("audio") or (isinstance(content, str) and len(content) > 80):
                try:
                    return _decode_audio_delta(content) if isinstance(content, str) else (b"", 24000)
                except Exception:
                    continue
        audio = chunk.get("audio") or {}
        if isinstance(audio, dict) and audio.get("data"):
            try:
                return _decode_audio_delta(audio["data"])
            except Exception:
                return b"", 24000
        return b"", 24000

    async def pull_audio(self) -> list[tuple[bytes, int]]:
        out: list[tuple[bytes, int]] = []
        while True:
            try:
                out.append(self._out.get_nowait())
            except asyncio.QueueEmpty:
                break
        return out

    async def interrupt(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            trace_stream("speech", "qwen.cancel", reason="interrupt")
        self._busy = False
        self._utt.clear()
        while not self._out.empty():
            try:
                self._out.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def close(self) -> None:
        await self.interrupt()
        if self._session is not None:
            await self._session.close()
            self._session = None
