"""Resample, RMS, leading-silence trim, and a jitter-buffered LiveKit playout clock.

Jitter buffer design matches the Runtype Persona AudioWorklet
(DEFAULT_PREBUFFER_MS=150) and the OpenBMB MiniCPM-o Demo
(playback_delay_ms=200).  MiniCPM-o emits ~1 s PCM chunks with
200-400 ms gaps between them; HiFT eager-fallback on vLLM-Omni 0.30.0
can pause 5-30 s between deltas (#5069 / #5657 warmup still pending).

The prebuffer absorbs normal inter-chunk jitter. PLAYOUT_HOLD_S only keeps
the transport clock warm briefly; longer model stalls become true idle so
VAD does not mistake user speech during dead air for a barge-in.
"""

from __future__ import annotations

import array
import asyncio
import logging
import math
import os
import time
from collections.abc import Awaitable, Callable

from trace import RateLimiter, hires_enabled, hires_period, trace_stream

LOG = logging.getLogger("audio")

WEBRTC_RATE = 48000
WEBRTC_CHANNELS = 1
WEBRTC_FRAME_SAMPLES = WEBRTC_RATE // 50
WEBRTC_FRAME_BYTES = WEBRTC_FRAME_SAMPLES * 2 * WEBRTC_CHANNELS
MODEL_RATE = 16000
FRAME_S = 0.020

# --- Playout jitter buffer (env-tunable) ---
# Prebuffer before the first frame of a new turn.  Runtype Persona
# uses 150 ms for realtime server-paced audio; OpenBMB Demo uses 200 ms.
PLAYOUT_START_MS = float(os.environ.get("PLAYOUT_START_MS", "150"))
# Rebuffer after an underrun.  Runtype uses same as start (150).
PLAYOUT_REBUFFER_MS = float(os.environ.get("PLAYOUT_REBUFFER_MS", "150"))
# Safety cap: max ms the prebuffer gate can hold before flushing anyway.
# A short first/final delta may never reach the 150 ms target. Never hold it
# for seconds while the model is stalled; release after one small jitter span.
PLAYOUT_MAX_WAIT_MS = float(os.environ.get("PLAYOUT_MAX_WAIT_MS", "300"))
# Briefly keep the LiveKit clock warm after real audio drains. Longer model
# stalls are not bridgeable with silence and must not count as bot playback.
HOLD_SILENCE_S = float(os.environ.get("PLAYOUT_HOLD_S", "0.25"))
BOT_AUDIBLE_GRACE_MS = float(os.environ.get("BOT_AUDIBLE_GRACE_MS", "120"))

CAPTURE_TIMEOUT_S = float(os.environ.get("PLAYOUT_CAPTURE_TIMEOUT_S", "0.08"))
STATS_EVERY_S = hires_period(5.0) if not hires_enabled() else 0.5
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
    """48 kHz ring + 20 ms ticker with jitter-buffered prebuffer gate.

    Producer (playback loop) calls push(); the 20 ms clock calls take_frame()
    which either returns real audio or silence (during prebuffer / underrun
    hold) or None (idle — no audio pending).

    The prebuffer gate holds real audio and feeds silence until the ring
    reaches the waterline, then releases.  On underrun it re-enters buffering
    at the rebuffer waterline.  This matches the Runtype Persona AudioWorklet
    pattern (waterline → play → underrun → re-buffer → play).
    """

    def __init__(self) -> None:
        self._ring = bytearray()
        self._resampler: LinearResampler | None = None
        self._last_push = 0.0
        self._last_real_frame = 0.0
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
        self._underrun_open = False
        self._first_frame = False
        # Jitter buffer state.  Start "buffering": hold real audio until the
        # ring reaches the prebuffer target, feeding silence to keep the clock
        # warm.
        self._buffering = True
        self._buf_target_ms = PLAYOUT_START_MS
        self._buffering_t0 = 0.0
        # Set each tick by the playback loop: True while the model is producing.
        # When it goes False we flush whatever is buffered (short-reply tail).
        self.source_active = False
        # Per-turn diagnostics (bounded by idle / clear).
        self._turn_open = False
        self._turn_frames = 0
        self._turn_silence = 0
        self._underrun_episodes = 0
        self._underrun_total_ms = 0.0
        self._longest_underrun_ms = 0.0
        self._ring_ms_min = -1.0
        self._underrun_t0 = 0.0
        self._underrun_frames = 0
        self.underruns_total = 0
        # Clock-slip diagnostics.
        self._slip_max_ms = 0.0
        self._late_ticks = 0
        self._slip_rl = RateLimiter(2.0)
        self._capto_rl = RateLimiter(2.0)
        self._ring_sample_rl = RateLimiter(0.1)
        self._buffering_rl = RateLimiter(0.1)
        self._push_rl = RateLimiter(1.0)

    @property
    def ring_ms(self) -> float:
        return 1000.0 * (len(self._ring) / 2.0) / WEBRTC_RATE

    @property
    def audible(self) -> bool:
        """True only while real bot audio is playing or queued for playout."""
        now = time.monotonic()
        queued = bool(self._ring) and not self._buffering
        recent = (
            self._last_real_frame > 0.0
            and (now - self._last_real_frame) * 1000.0 <= BOT_AUDIBLE_GRACE_MS
        )
        return queued or recent

    def push(self, pcm: bytes, sample_rate: int) -> None:
        if not pcm:
            return
        rate = int(sample_rate) or WEBRTC_RATE
        first = self._first_push
        if self._first_push:
            pcm = trim_leading_silence(pcm)
            self._first_push = False
        if rate != WEBRTC_RATE:
            if self._resampler is None or self._resampler.dst_rate != WEBRTC_RATE:
                self._resampler = LinearResampler(WEBRTC_RATE)
            pcm = self._resampler.feed(pcm, rate)
            if not pcm:
                return
        self._ring.extend(pcm)
        self._last_push = time.monotonic()
        self._active = True
        if first or hires_enabled() or self._push_rl.due():
            dur_ms = int(1000.0 * (len(pcm) / 2.0) / WEBRTC_RATE)
            trace_stream(
                "speech",
                "playout.push",
                bytes=len(pcm),
                rate=rate,
                ring_ms=round(self.ring_ms, 1),
                dur_ms=dur_ms,
                first=first,
            )

    def _reset_turn_stats(self) -> None:
        self._turn_frames = 0
        self._turn_silence = 0
        self._underrun_episodes = 0
        self._underrun_total_ms = 0.0
        self._longest_underrun_ms = 0.0
        self._ring_ms_min = -1.0
        self._underrun_open = False
        self._underrun_t0 = 0.0
        self._underrun_frames = 0

    def _emit_turn_summary(self, reason: str) -> None:
        if self._turn_frames <= 0:
            return
        trace_stream(
            "speech",
            "playout.turn_summary",
            reason=reason,
            frames=self._turn_frames,
            silence_frames=self._turn_silence,
            underrun_episodes=self._underrun_episodes,
            underrun_total_ms=int(self._underrun_total_ms),
            longest_underrun_ms=int(self._longest_underrun_ms),
            ring_ms_min=round(self._ring_ms_min, 1) if self._ring_ms_min >= 0 else -1,
        )

    def clear(self) -> None:
        if self._turn_open:
            self._emit_turn_summary("clear")
            self._turn_open = False
        self._reset_turn_stats()
        self._ring.clear()
        self._resampler = None
        self._active = False
        self._last_push = 0.0
        self._last_real_frame = 0.0
        self._first_push = True
        self.clears += 1
        self._underrun_open = False
        self._first_frame = False
        self._buffering = True
        self._buf_target_ms = PLAYOUT_START_MS
        self._buffering_t0 = 0.0
        trace_stream("speech", "playout.clear", clears=self.clears)

    def take_frame(self, now: float | None = None) -> bytes | None:
        now = time.monotonic() if now is None else now
        need = WEBRTC_FRAME_BYTES

        if hires_enabled() and self._ring_sample_rl.due():
            trace_stream(
                "speech",
                "playout.ring_sample",
                ring_ms=round(self.ring_ms, 1),
                buffering=self._buffering,
                source_active=self.source_active,
                active=self._active,
            )

        # --- Jitter buffer gate: while buffering, hold real audio and feed
        # silence until the ring reaches the target (or the source stops, or
        # we hit the safety cap).  This is what makes playback smooth. ---
        if self._buffering and self._ring:
            fill_ms = self.ring_ms
            if self._buffering_t0 == 0.0:
                self._buffering_t0 = now
            waited_ms = (now - self._buffering_t0) * 1000.0
            ready = (
                fill_ms >= self._buf_target_ms
                or not self.source_active
                or waited_ms >= PLAYOUT_MAX_WAIT_MS
            )
            if not ready:
                self.silence_frames += 1
                if hires_enabled() and self._buffering_rl.due():
                    trace_stream(
                        "speech",
                        "playout.buffering",
                        fill_ms=round(fill_ms, 1),
                        target_ms=int(self._buf_target_ms),
                        waited_ms=int(waited_ms),
                    )
                return self._silence
            self._buffering = False
            self._buffering_t0 = 0.0
            trace_stream(
                "speech",
                "playout.resume",
                ring_ms=round(fill_ms, 1),
                target_ms=int(self._buf_target_ms),
                waited_ms=int(waited_ms),
                source_active=self.source_active,
            )

        if len(self._ring) >= need:
            frame = bytes(self._ring[:need])
            del self._ring[:need]
            self.frames += 1
            self._active = True
            if not self._turn_open:
                self._turn_open = True
                self._reset_turn_stats()
            self._turn_frames += 1
            self._last_real_frame = now
            rm = self.ring_ms
            if self._ring_ms_min < 0 or rm < self._ring_ms_min:
                self._ring_ms_min = rm
            if self._underrun_open:
                self._underrun_open = False
                episode_ms = (now - self._underrun_t0) * 1000.0 if self._underrun_t0 else 0.0
                self._underrun_total_ms += episode_ms
                if episode_ms > self._longest_underrun_ms:
                    self._longest_underrun_ms = episode_ms
                trace_stream(
                    "speech",
                    "playout.recover",
                    ring_ms=round(rm, 1),
                    underrun_ms=int(episode_ms),
                    silence_frames_episode=self._underrun_frames,
                )
            if not self._first_frame:
                self._first_frame = True
                trace_stream("speech", "playout.first_frame", bytes=len(frame), ring_ms=round(rm, 1))
            return frame

        if self._ring:
            frame = bytes(self._ring) + self._silence[len(self._ring) :]
            self._ring.clear()
            self.frames += 1
            if not self._turn_open:
                self._turn_open = True
                self._reset_turn_stats()
            self._turn_frames += 1
            self._last_real_frame = now
            return frame[:need]

        if self._active and self._last_push and (now - self._last_push) <= HOLD_SILENCE_S:
            self.silence_frames += 1
            self._turn_silence += 1
            if not self._underrun_open:
                self._underrun_open = True
                self._underrun_episodes += 1
                self.underruns_total += 1
                self._underrun_t0 = now
                self._underrun_frames = 0
                # Enter re-buffer mode so the next batch of audio accumulates
                # before we start draining again.
                if not self._buffering:
                    self._buffering = True
                    self._buf_target_ms = PLAYOUT_REBUFFER_MS
                    self._buffering_t0 = now
                trace_stream(
                    "speech",
                    "playout.underrun",
                    hold=True,
                    since_push_ms=int((now - self._last_push) * 1000),
                    episode=self._underrun_episodes,
                )
            self._underrun_frames += 1
            return self._silence

        if self._active:
            trace_stream(
                "speech",
                "playout.idle",
                since_push_ms=int((now - self._last_push) * 1000) if self._last_push else -1,
            )
            if self._turn_open:
                self._emit_turn_summary("idle")
                self._turn_open = False
        self._active = False
        self._underrun_open = False
        self.idle_ticks += 1
        return None

    def _emit_stats(self) -> None:
        now = time.monotonic()
        if now - self._stats_t < STATS_EVERY_S:
            return
        self._stats_t = now
        underrun = self.silence_frames > 0
        trace_stream(
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
            underrun_episodes=self._underrun_episodes,
            slip_max_ms=int(self._slip_max_ms),
            late_ticks=self._late_ticks,
        )
        self.frames = 0
        self.silence_frames = 0
        self.idle_ticks = 0
        self.capture_timeouts = 0
        self.clock_slips = 0
        self._slip_max_ms = 0.0
        self._late_ticks = 0

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
                        if self._capto_rl.due():
                            trace_stream(
                                "speech",
                                "playout.capture_timeout",
                                ms=int(CAPTURE_TIMEOUT_S * 1000),
                            )
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        LOG.debug("playout capture skipped", exc_info=True)
                    else:
                        block_ms = int((time.monotonic() - t0) * 1000)
                        if block_ms >= 50:
                            trace_stream("speech", "playout.block", block_ms=block_ms, where="capture")
                self._emit_stats()
                next_t += FRAME_S
                delay = next_t - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    late_ms = -delay * 1000.0
                    if late_ms > self._slip_max_ms:
                        self._slip_max_ms = late_ms
                    self._late_ticks += 1
                    if self._slip_rl.due():
                        trace_stream(
                            "speech",
                            "playout.clock_slip",
                            late_ms=int(late_ms),
                            late_ticks=self._late_ticks,
                        )
                    if delay < -0.10:
                        self.clock_slips += 1
                        next_t = time.monotonic()
        except asyncio.CancelledError:
            raise
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False
        self.clear()
