"""Resample, RMS, leading-silence trim, and a 20 ms LiveKit playout clock."""

from __future__ import annotations

import array
import asyncio
import logging
import math
import os
import time
from collections.abc import Awaitable, Callable

from trace import trace

LOG = logging.getLogger("audio")

WEBRTC_RATE = 48000
WEBRTC_CHANNELS = 1
WEBRTC_FRAME_SAMPLES = WEBRTC_RATE // 50
WEBRTC_FRAME_BYTES = WEBRTC_FRAME_SAMPLES * 2 * WEBRTC_CHANNELS
MODEL_RATE = 16000
FRAME_S = 0.020
HOLD_SILENCE_S = float(os.environ.get("PLAYOUT_HOLD_S", "0.40"))
CAPTURE_TIMEOUT_S = float(os.environ.get("PLAYOUT_CAPTURE_TIMEOUT_S", "0.08"))
STATS_EVERY_S = 5.0
SILENCE_RMS = int(os.environ.get("LEADING_SILENCE_RMS", "80"))


class LinearResampler:
    def __init__(self, dst_rate: int) -> None:
        self.dst_rate = dst_rate
        self.src_rate: int | None = None
        self._buf = array.array("h")
        self._pos = 0.0

    def feed(self, pcm: bytes, src_rate: int) -> bytes:
        if src_rate != self.src_rate:
            self.src_rate = src_rate
            self._buf = array.array("h")
            self._pos = 0.0
        if src_rate == self.dst_rate:
            return pcm
        n = (len(pcm) // 2) * 2
        incoming = array.array("h")
        incoming.frombytes(pcm[:n])
        self._buf.extend(incoming)
        if len(self._buf) < 2:
            return b""
        ratio = self.src_rate / self.dst_rate
        out = array.array("h")
        buflen = len(self._buf)
        while int(self._pos) + 1 < buflen:
            i = int(self._pos)
            frac = self._pos - i
            s0 = self._buf[i]
            s1 = self._buf[i + 1]
            out.append(int(s0 + (s1 - s0) * frac))
            self._pos += ratio
        consumed = int(self._pos)
        if consumed > 0:
            del self._buf[:consumed]
            self._pos -= consumed
        return out.tobytes()


def pcm_rms(pcm16: bytes) -> int:
    if len(pcm16) < 2:
        return 0
    n = len(pcm16) // 2
    samples = array.array("h")
    samples.frombytes(pcm16[: n * 2])
    if not samples:
        return 0
    acc = 0
    for s in samples:
        acc += s * s
    return int(math.sqrt(acc / len(samples)))


def downmix_to_mono(pcm: bytes, channels: int) -> bytes:
    if channels <= 1 or not pcm:
        return pcm
    n = (len(pcm) // 2) // channels
    if n <= 0:
        return pcm
    src = array.array("h")
    src.frombytes(pcm[: n * channels * 2])
    out = array.array("h")
    for i in range(n):
        acc = 0
        base = i * channels
        for c in range(channels):
            acc += src[base + c]
        out.append(int(acc / channels))
    return out.tobytes()


def trim_leading_silence(pcm: bytes, rms_floor: int = SILENCE_RMS, win: int = 320) -> bytes:
    """Drop leading windows whose RMS is below *rms_floor*. win is samples."""
    if len(pcm) < win * 2:
        return pcm
    samples = array.array("h")
    samples.frombytes(pcm[: (len(pcm) // 2) * 2])
    i = 0
    n = len(samples)
    while i + win <= n:
        acc = 0
        for s in samples[i : i + win]:
            acc += s * s
        rms = int(math.sqrt(acc / win))
        if rms >= rms_floor:
            break
        i += win
    if i <= 0:
        return pcm
    return samples[i:].tobytes()


class ClockedPlayout:
    """48 kHz ring + 20 ms ticker. Producer never awaits LiveKit."""

    def __init__(self) -> None:
        self._ring = bytearray()
        self._resampler: LinearResampler | None = None
        self._last_push = 0.0
        self._active = False
        self._running = False
        self.frames = 0
        self.silence_frames = 0
        self.idle_ticks = 0
        self.capture_timeouts = 0
        self.clock_slips = 0
        self.clears = 0
        self._stats_t = time.monotonic()
        self._silence = b"\x00" * WEBRTC_FRAME_BYTES
        self._first_push = True

    @property
    def ring_ms(self) -> float:
        return 1000.0 * (len(self._ring) / 2.0) / WEBRTC_RATE

    def push(self, pcm: bytes, sample_rate: int) -> None:
        if not pcm:
            return
        rate = int(sample_rate) or WEBRTC_RATE
        if self._first_push:
            pcm = trim_leading_silence(pcm)
            self._first_push = False
            trace("speech", "playout.push", bytes=len(pcm), rate=rate, first=True)
        if rate != WEBRTC_RATE:
            if self._resampler is None or self._resampler.dst_rate != WEBRTC_RATE:
                self._resampler = LinearResampler(WEBRTC_RATE)
            pcm = self._resampler.feed(pcm, rate)
            if not pcm:
                return
        self._ring.extend(pcm)
        self._last_push = time.monotonic()
        self._active = True

    def clear(self) -> None:
        self._ring.clear()
        self._resampler = None
        self._active = False
        self._last_push = 0.0
        self._first_push = True
        self.clears += 1
        trace("speech", "playout.clear", clears=self.clears)

    def take_frame(self, now: float | None = None) -> bytes | None:
        now = time.monotonic() if now is None else now
        need = WEBRTC_FRAME_BYTES
        if len(self._ring) >= need:
            frame = bytes(self._ring[:need])
            del self._ring[:need]
            self.frames += 1
            self._active = True
            return frame
        if self._ring:
            frame = bytes(self._ring) + self._silence[len(self._ring) :]
            self._ring.clear()
            self.frames += 1
            return frame[:need]
        if self._active and self._last_push and (now - self._last_push) <= HOLD_SILENCE_S:
            self.silence_frames += 1
            return self._silence
        self._active = False
        self.idle_ticks += 1
        return None

    def _emit_stats(self) -> None:
        now = time.monotonic()
        if now - self._stats_t < STATS_EVERY_S:
            return
        self._stats_t = now
        underrun = self.silence_frames > 0
        trace(
            "speech",
            "playout.underrun" if underrun else "playout.stats",
            frames=self.frames,
            silence_frames=self.silence_frames,
            idle_ticks=self.idle_ticks,
            ring_ms=round(self.ring_ms, 1),
            capture_timeouts=self.capture_timeouts,
            clock_slips=self.clock_slips,
            clears=self.clears,
            underrun=underrun,
        )
        self.frames = 0
        self.silence_frames = 0
        self.idle_ticks = 0
        self.capture_timeouts = 0
        self.clock_slips = 0

    async def run(self, capture: Callable[[bytes, int], Awaitable[None]]) -> None:
        self._running = True
        next_t = time.monotonic()
        try:
            while self._running:
                frame = self.take_frame()
                if frame is not None:
                    t0 = time.monotonic()
                    try:
                        await asyncio.wait_for(capture(frame, WEBRTC_RATE), timeout=CAPTURE_TIMEOUT_S)
                    except TimeoutError:
                        self.capture_timeouts += 1
                        trace("speech", "playout.block", block_ms=int(CAPTURE_TIMEOUT_S * 1000))
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        LOG.debug("playout capture skipped", exc_info=True)
                    else:
                        block_ms = int((time.monotonic() - t0) * 1000)
                        if block_ms >= 50:
                            trace("speech", "playout.block", block_ms=block_ms)
                self._emit_stats()
                next_t += FRAME_S
                delay = next_t - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                elif delay < -0.10:
                    self.clock_slips += 1
                    next_t = time.monotonic()
        except asyncio.CancelledError:
            raise
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False
        self.clear()
