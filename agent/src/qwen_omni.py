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
from trace import trace, trace_exc, trace_turn

LOG = logging.getLogger("qwen")


def _base_url() -> str:
    return (os.environ.get("MODEL_BASE_URL") or "http://127.0.0.1:8099").rstrip("/")


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
        self._task: asyncio.Task | None = None

    @property
    def generating(self) -> bool:
        return self._busy

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
            trace("speech", "qwen.ready_retry", attempt=attempt, error=last_err, url=url)
            await asyncio.sleep(min(delay, 15.0))
            delay = min(delay * 1.25, 15.0)
        else:
            raise RuntimeError(f"model-server never ready at {url}: {last_err}")
        trace("speech", "voice.ref_loaded", source=f"qwen_speaker:{self.speaker}")

    async def push_audio(self, pcm16: bytes, sample_rate: int) -> None:
        if not pcm16:
            return
        if sample_rate != MODEL_RATE:
            if self._in_rs is None:
                self._in_rs = LinearResampler(MODEL_RATE)
            pcm16 = self._in_rs.feed(pcm16, sample_rate)
            if not pcm16:
                return
        self._utt.extend(pcm16)

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
            trace("speech", "qwen.request", skipped="too_short", bytes=len(pcm))
            return
        self._busy = True
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
        with trace_turn("speech", event="qwen", bytes=len(pcm), jpeg=bool(jpeg), speaker=self.speaker):
            try:
                async with self._session.post(url, json=body) as resp:
                    if resp.status >= 400:
                        text = await resp.text()
                        trace("speech", "error.qwen_http", status=resp.status, body=text[:300])
                        return
                    async for raw in resp.content:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        pcm_out, rate = self._extract_audio(chunk)
                        if not pcm_out:
                            continue
                        audio_bytes += len(pcm_out)
                        if first:
                            first = False
                            trace(
                                "speech",
                                "qwen.first_audio",
                                latency_ms=int((time.monotonic() - t0) * 1000),
                                bytes=len(pcm_out),
                                rate=rate,
                            )
                        await self._out.put((pcm_out, rate))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                trace_exc("speech", "error.qwen_stream", exc)
            finally:
                self._busy = False
                trace(
                    "speech",
                    "qwen.done",
                    latency_ms=int((time.monotonic() - t0) * 1000),
                    audio_bytes=audio_bytes,
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
