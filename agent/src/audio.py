"""Resample, RMS, leading-silence trim, and a jitter-buffered LiveKit playout clock.

Jitter buffer design matches the Runtype Persona AudioWorklet
(DEFAULT_PREBUFFER_MS=150) and the OpenBMB MiniCPM-o Demo
(playback_delay_ms=200).  MiniCPM-o emits ~1 s PCM chunks with
200-400 ms gaps between them; HiFT eager-fallback on vLLM-Omni 0.30.0
can pause 5-30 s between deltas (#5069 / #5657 warmup still pending).

While the model is generating, empty-ring gaps are silence (no rebuffer).
PLAYOUT_HOLD_S only covers the tail after generation stops. Longer dead
air then goes idle so VAD can treat it as the user's turn.
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
# Safety cap once we have more than the first vocoder blob.
PLAYOUT_MAX_WAIT_MS = float(os.environ.get("PLAYOUT_MAX_WAIT_MS", "300"))
# Qwen's first delta is always ~217 ms, then 0.3–1.8 s of vocoder think,
# then the body. Starting on the preroll guarantees a mid-sentence hole.
# Hold that first blob until blob 2 arrives (or this timeout).
PLAYOUT_FIRST_BLOB_WAIT_MS = float(os.environ.get("PLAYOUT_FIRST_BLOB_WAIT_MS", "2500"))
PLAYOUT_FIRST_BLOB_MIN_MS = float(os.environ.get("PLAYOUT_FIRST_BLOB_MIN_MS", "400"))
# Briefly keep the LiveKit clock warm after real audio drains. Longer model
# stalls are not bridgeable with silence and must not count as bot playback.
HOLD_SILENCE_S = float(os.environ.get("PLAYOUT_HOLD_S", "0.40"))
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

    The prebuffer gate holds the first vocoder blob until a second packet
    arrives (Qwen preroll is ~217 ms; starting it guarantees a hole).
    Gaps after playback has started, while the model is still generating,
    are silence rather than a rebuffer.
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
        self.frame_seq = 0
        self.push_seq = 0
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
        self._turn_pushes = 0
        self._first_blob_t = 0.0
        self._first_blob_dur_ms = 0
        self._logged_first_blob_hold = False
        self._gen_gap_open = False
        self._gen_gap_t0 = 0.0
        self._gen_gap_frames = 0
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
        # Prebuffer wait tracking for session lab
        self._turn_prebuffer_ms = -1.0
        self._prebuffer_waiting = False
        self._prebuffer_t0 = 0.0
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

    def _arm_new_turn(self) -> None:
        """Reset the jitter gate so a later turn does not inherit stale wait."""
        now = time.monotonic()
        if self._last_real_frame:
            trace_stream(
                "speech",
                "playout.turn_gap",
                since_real_ms=int((now - self._last_real_frame) * 1000),
            )
        self._buffering = True
        self._buf_target_ms = PLAYOUT_START_MS
        self._buffering_t0 = 0.0
        self._turn_pushes = 0
        self._first_blob_t = 0.0
        self._first_blob_dur_ms = 0
        self._logged_first_blob_hold = False
        self._gen_gap_open = False
        self._gen_gap_t0 = 0.0
        self._gen_gap_frames = 0
        self._first_frame = False
        self._first_push = True
        self._underrun_open = False

    def _note_gen_gap_close(self, now: float) -> None:
        if not self._gen_gap_open:
            return
        frames = self._gen_gap_frames
        gap_ms = int((now - self._gen_gap_t0) * 1000) if self._gen_gap_t0 else 0
        self._gen_gap_open = False
        self._gen_gap_t0 = 0.0
        self._gen_gap_frames = 0
        # One event per hole, and only if it was long enough to hear.
        if frames >= 5:
            trace_stream(
                "speech",
                "playout.sentence_gap",
                gap_ms=gap_ms,
                silence_frames=frames,
                where="during_generate",
            )

    def push(self, pcm: bytes, sample_rate: int) -> None:
        if not pcm:
            return
        if not self._active:
            self._arm_new_turn()
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
        now = time.monotonic()
        self._last_push = now
        self._active = True
        self.push_seq += 1
        self._turn_pushes += 1
        dur_ms = int(1000.0 * (len(pcm) / 2.0) / WEBRTC_RATE)
        if self._turn_pushes == 1:
            self._first_blob_t = now
            self._first_blob_dur_ms = dur_ms
        elif self._turn_pushes == 2:
            gap_ms = int((now - self._first_blob_t) * 1000) if self._first_blob_t else -1
            trace_stream(
                "speech",
                "playout.blob_gap",
                gap_ms=gap_ms,
                first_dur_ms=self._first_blob_dur_ms,
                second_dur_ms=dur_ms,
                ring_ms=round(self.ring_ms, 1),
                source_active=self.source_active,
                buffering=self._buffering,
            )
        if first or self._turn_pushes <= 2 or hires_enabled() or self._push_rl.due():
            trace_stream(
                "speech",
                "playout.push",
                packet_seq=self.push_seq,
                turn_pushes=self._turn_pushes,
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
        # Keep _turn_prebuffer_ms: first real frame calls this after the
        # prebuffer gate has already recorded the wait for this turn.

    def _trace_frame(
        self,
        kind: str,
        frame: bytes,
        *,
        ring_before_ms: float,
        real_bytes: int,
        now: float,
    ) -> None:
        if not hires_enabled():
            return
        self.frame_seq += 1
        trace_stream(
            "speech",
            "playout.frame",
            frame_seq=self.frame_seq,
            kind=kind,
            bytes=len(frame),
            real_bytes=real_bytes,
            rms=pcm_rms(frame) if real_bytes else 0,
            ring_before_ms=round(ring_before_ms, 1),
            ring_ms=round(self.ring_ms, 1),
            buffering=self._buffering,
            source_active=self.source_active,
            audible=self.audible,
            since_push_ms=int((now - self._last_push) * 1000) if self._last_push else -1,
        )

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
            prebuffer_ms=round(self._turn_prebuffer_ms, 1) if self._turn_prebuffer_ms >= 0 else -1,
        )

    def _close_turn(self, reason: str) -> None:
        if self._turn_open:
            self._emit_turn_summary(reason)
            self._turn_open = False
        self._turn_prebuffer_ms = -1.0
        self._prebuffer_waiting = False
        self._prebuffer_t0 = 0.0

    def _begin_turn(self) -> None:
        if self._turn_open:
            return
        saved_prebuffer = self._turn_prebuffer_ms
        self._turn_open = True
        self._reset_turn_stats()
        self._turn_prebuffer_ms = saved_prebuffer

    def turn_stats(self) -> dict[str, float | int]:
        """Return turn-level playout stats for session lab."""
        return {
            "prebuffer_ms": round(self._turn_prebuffer_ms, 1) if self._turn_prebuffer_ms >= 0 else -1.0,
            "underrun_episodes": self._underrun_episodes,
            "underrun_total_ms": round(self._underrun_total_ms, 1),
            "longest_underrun_ms": round(self._longest_underrun_ms, 1),
            "ring_ms_min": round(self._ring_ms_min, 1) if self._ring_ms_min >= 0 else -1.0,
            "turn_frames": self._turn_frames,
            "turn_silence": self._turn_silence,
            "slip_max_ms": round(self._slip_max_ms, 1),
        }

    def clear(self) -> None:
        self._close_turn("clear")
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
        self._turn_pushes = 0
        self._first_blob_t = 0.0
        self._first_blob_dur_ms = 0
        self._logged_first_blob_hold = False
        self._gen_gap_open = False
        self._gen_gap_t0 = 0.0
        self._gen_gap_frames = 0
        trace_stream("speech", "playout.clear", clears=self.clears)

    def take_frame(self, now: float | None = None) -> bytes | None:
        now = time.monotonic() if now is None else now
        need = WEBRTC_FRAME_BYTES
        ring_before_ms = self.ring_ms

        if hires_enabled() and self._ring_sample_rl.due():
            trace_stream(
                "speech",
                "playout.ring_sample",
                ring_ms=round(self.ring_ms, 1),
                buffering=self._buffering,
                source_active=self.source_active,
                active=self._active,
            )

        # --- Jitter buffer gate. Qwen's first delta is ~217 ms; starting it
        # before blob 2 arrives is the between-sentence hole in Loki. ---
        if self._buffering and self._ring:
            fill_ms = self.ring_ms
            if self._buffering_t0 == 0.0:
                self._buffering_t0 = now
            if not self._prebuffer_waiting:
                self._prebuffer_waiting = True
                self._prebuffer_t0 = now
            waited_ms = (now - self._buffering_t0) * 1000.0
            first_blob = (
                self._turn_pushes < 2
                and fill_ms < PLAYOUT_FIRST_BLOB_MIN_MS
                and self.source_active
            )
            reason = "prebuffer"
            if first_blob:
                ready = waited_ms >= PLAYOUT_FIRST_BLOB_WAIT_MS
                reason = "first_blob_timeout" if ready else "wait_second_blob"
            else:
                ready = (
                    fill_ms >= self._buf_target_ms
                    or not self.source_active
                    or waited_ms >= PLAYOUT_MAX_WAIT_MS
                )
                if ready:
                    if not self.source_active:
                        reason = "source_done"
                    elif fill_ms >= self._buf_target_ms:
                        reason = "fill"
                    else:
                        reason = "max_wait"
            if not ready:
                self.silence_frames += 1
                if reason == "wait_second_blob" and not self._logged_first_blob_hold:
                    self._logged_first_blob_hold = True
                    trace_stream(
                        "speech",
                        "playout.hold_first_blob",
                        fill_ms=round(fill_ms, 1),
                        waited_ms=int(waited_ms),
                        pushes=self._turn_pushes,
                        min_ms=int(PLAYOUT_FIRST_BLOB_MIN_MS),
                    )
                elif hires_enabled() and self._buffering_rl.due():
                    trace_stream(
                        "speech",
                        "playout.buffering",
                        fill_ms=round(fill_ms, 1),
                        target_ms=int(self._buf_target_ms),
                        waited_ms=int(waited_ms),
                        reason=reason,
                    )
                self._trace_frame(
                    "prebuffer_silence",
                    self._silence,
                    ring_before_ms=ring_before_ms,
                    real_bytes=0,
                    now=now,
                )
                return self._silence
            self._buffering = False
            self._buffering_t0 = 0.0
            if self._prebuffer_waiting:
                if self._turn_prebuffer_ms < 0:
                    self._turn_prebuffer_ms = (now - self._prebuffer_t0) * 1000.0
                self._prebuffer_waiting = False
            trace_stream(
                "speech",
                "playout.resume",
                ring_ms=round(fill_ms, 1),
                target_ms=int(self._buf_target_ms),
                waited_ms=int(waited_ms),
                prebuffer_ms=round(self._turn_prebuffer_ms, 1),
                source_active=self.source_active,
                reason=reason,
                turn_pushes=self._turn_pushes,
            )

        if len(self._ring) >= need:
            frame = bytes(self._ring[:need])
            del self._ring[:need]
            self.frames += 1
            self._active = True
            self._begin_turn()
            self._turn_frames += 1
            self._last_real_frame = now
            rm = self.ring_ms
            if self._ring_ms_min < 0 or rm < self._ring_ms_min:
                self._ring_ms_min = rm
            self._note_gen_gap_close(now)
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
                trace_stream(
                    "speech",
                    "playout.first_frame",
                    bytes=len(frame),
                    ring_ms=round(rm, 1),
                    turn_pushes=self._turn_pushes,
                    prebuffer_ms=round(self._turn_prebuffer_ms, 1),
                )
            self._trace_frame(
                "real",
                frame,
                ring_before_ms=ring_before_ms,
                real_bytes=len(frame),
                now=now,
            )
            return frame

        if self._ring:
            real_bytes = len(self._ring)
            frame = bytes(self._ring) + self._silence[real_bytes:]
            self._ring.clear()
            self.frames += 1
            self._begin_turn()
            self._turn_frames += 1
            self._last_real_frame = now
            self._note_gen_gap_close(now)
            self._trace_frame(
                "real_padded",
                frame[:need],
                ring_before_ms=ring_before_ms,
                real_bytes=real_bytes,
                now=now,
            )
            return frame[:need]

        # MiniCPM body chunks are ~960 ms with ~0.5-1 s vocoder gaps while
        # generating. Rebuffer-on-underrun was a hitch on every chunk
        # (Loki 00:04: 17 underruns / 5 replies). Keep the clock warm and
        # play the next delta immediately.
        if self._active and self.source_active:
            self.silence_frames += 1
            self._turn_silence += 1
            if not self._gen_gap_open:
                self._gen_gap_open = True
                self._gen_gap_t0 = now
                self._gen_gap_frames = 0
            self._gen_gap_frames += 1
            self._trace_frame(
                "gap_silence",
                self._silence,
                ring_before_ms=ring_before_ms,
                real_bytes=0,
                now=now,
            )
            return self._silence

        if self._active and self._last_push and (now - self._last_push) <= HOLD_SILENCE_S:
            self.silence_frames += 1
            self._turn_silence += 1
            if not self._underrun_open:
                self._underrun_open = True
                self._underrun_episodes += 1
                self.underruns_total += 1
                self._underrun_t0 = now
                self._underrun_frames = 0
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
            self._trace_frame(
                "underrun_silence",
                self._silence,
                ring_before_ms=ring_before_ms,
                real_bytes=0,
                now=now,
            )
            return self._silence

        if self._active:
            self._note_gen_gap_close(now)
            trace_stream(
                "speech",
                "playout.idle",
                since_push_ms=int((now - self._last_push) * 1000) if self._last_push else -1,
            )
            self._close_turn("idle")
        self._active = False
        self._underrun_open = False
        self.idle_ticks += 1
        return None

    def _emit_stats(self) -> None:
        now = time.monotonic()
        if now - self._stats_t < STATS_EVERY_S:
            return
        self._stats_t = now
        trace_stream(
            "speech",
            "playout.stats",
            frames=self.frames,
            silence_frames=self.silence_frames,
            idle_ticks=self.idle_ticks,
            ring_ms=round(self.ring_ms, 1),
            capture_timeouts=self.capture_timeouts,
            clock_slips=self.clock_slips,
            clears=self.clears,
            underrun_episodes=self._underrun_episodes,
            gen_gap_open=self._gen_gap_open,
            buffering=self._buffering,
            turn_pushes=self._turn_pushes,
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
