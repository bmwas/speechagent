"""VAD + echo guard. MiniCPM-o 4.5 does not self-yield mid-turn.

``EnergyVad`` is RMS hangover. ``LiveKitSileroVad`` wraps LiveKit's bundled
``inference.VAD(model="silero")`` as a drop-in with the same ``feed()`` edges.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from typing import Any

from audio import MODEL_RATE, pcm_rms
from trace import STREAM_PERIOD_S, StreamMeter, hires_enabled, trace_stream

LOG = logging.getLogger("barge_in")

VAD_START_SECS = float(os.environ.get("VAD_START_SECS", "0.15"))
VAD_STOP_SECS = float(os.environ.get("VAD_STOP_SECS", "0.80"))
MIC_GATE_RMS = float(os.environ.get("VAD_MIN_VOLUME", "0.01")) * 32768.0
MIC_GATE_MIN = 80.0
MIC_GATE_MULT = 3.5
INTERRUPT_MIN_S = float(os.environ.get("VAD_INTERRUPT_MIN_S", "0.50"))


class EnergyVad:
    """RMS hangover VAD. Returns 'start' / 'stop' on edges, else None."""

    backend = "energy"

    def __init__(
        self,
        *,
        start_secs: float | None = None,
        stop_secs: float | None = None,
        silent: bool = False,
    ) -> None:
        self.start_secs = float(VAD_START_SECS if start_secs is None else start_secs)
        self.stop_secs = float(VAD_STOP_SECS if stop_secs is None else stop_secs)
        self.silent = silent
        self.speaking = False
        self.bot_playing = False
        self._noise_ema = 400.0
        self._loud_since: float | None = None
        self._quiet_since: float | None = None
        self._speech_start: float | None = None
        self.last_rms = 0
        self.frame_seq = 0
        self._stats = StreamMeter("speech", "vad.stats", period_s=STREAM_PERIOD_S)

    def _open_thresh(self) -> float:
        return max(MIC_GATE_MIN, self._noise_ema * MIC_GATE_MULT, MIC_GATE_RMS)

    def feed(self, pcm16: bytes) -> str | None:
        rms = pcm_rms(pcm16)
        self.last_rms = rms
        now = time.monotonic()
        open_t = self._open_thresh()
        self.frame_seq += 1
        if hires_enabled() and not self.silent:
            trace_stream(
                "speech",
                "vad.frame",
                frame_seq=self.frame_seq,
                bytes=len(pcm16),
                rms=rms,
                open_thresh=int(open_t),
                close_thresh=int(open_t * 0.6),
                speaking=self.speaking,
                bot_playing=self.bot_playing,
                noise_ema=int(self._noise_ema),
            )
        if not self.silent:
            self._stats.add(
                frames=1,
                rms=rms,
                open_thresh=int(open_t),
                speaking=int(self.speaking),
                bot_playing=int(self.bot_playing),
            )
        close_t = open_t * 0.6
        if not self.speaking and rms < open_t:
            self._noise_ema = 0.98 * self._noise_ema + 0.02 * float(rms)

        if rms >= open_t:
            if self._loud_since is None:
                self._loud_since = now
            self._quiet_since = None
            if not self.speaking and (now - self._loud_since) >= self.start_secs:
                if self.bot_playing:
                    held = now - self._loud_since
                    if held < INTERRUPT_MIN_S:
                        return None
                self.speaking = True
                self._speech_start = now
                if not self.silent:
                    trace_stream(
                        "speech",
                        "vad.start",
                        rms=int(rms),
                        open_thresh=int(open_t),
                        bot_playing=self.bot_playing,
                        noise_ema=int(self._noise_ema),
                    )
                return "start"
            return None

        self._loud_since = None
        if not self.speaking:
            return None
        if rms >= close_t:
            self._quiet_since = None
            return None
        if self._quiet_since is None:
            self._quiet_since = now
        if (now - self._quiet_since) >= self.stop_secs:
            dur = (now - self._speech_start) if self._speech_start else -1.0
            self.speaking = False
            self._speech_start = None
            self._quiet_since = None
            if not self.silent:
                trace_stream(
                    "speech",
                    "vad.stop",
                    rms=int(rms),
                    utterance_s=round(dur, 2),
                    bot_playing=self.bot_playing,
                )
            return "stop"
        return None


class LiveKitSileroVad:
    """Standalone LiveKit Silero VAD. Same duck-type as EnergyVad.

    ``feed()`` is synchronous: it pushes a 16 kHz frame and drains any edges
    the background stream has already emitted (one 32 ms inference window).
    Echo hold (``VAD_INTERRUPT_MIN_S``) still gates barge-in while the bot is
    audible — Silero has no ``bot_playing`` flag of its own.
    """

    backend = "silero"

    def __init__(
        self,
        *,
        start_secs: float | None = None,
        stop_secs: float | None = None,
    ) -> None:
        self.start_secs = float(VAD_START_SECS if start_secs is None else start_secs)
        self.stop_secs = float(VAD_STOP_SECS if stop_secs is None else stop_secs)
        self.speaking = False
        self.bot_playing = False
        self.last_rms = 0
        self.frame_seq = 0
        self._pending_start = False
        self._hold_until = 0.0
        self._speech_start: float | None = None
        self._edges: deque[str] = deque()
        self._stream: Any = None
        self._vad: Any = None
        self._task: asyncio.Task | None = None
        self._rtc: Any = None
        self._stats = StreamMeter("speech", "vad.stats", period_s=STREAM_PERIOD_S)
        # If Silero misses a turn (stream stall, conservative threshold), RMS
        # hangover still commits so "hello" is not dropped.
        self._energy = EnergyVad(start_secs=start_secs, stop_secs=stop_secs, silent=True)
        self._assist_active = False

    def apply_event(self, ev_type: str, *, probability: float = 0.0, now: float | None = None) -> str | None:
        """Process a Silero-style event. Used by the stream task and by unit tests."""
        del probability
        now = time.monotonic() if now is None else now
        kind = (ev_type or "").lower()
        if kind in {"start_of_speech", "start"}:
            if self.speaking or self._pending_start:
                return None
            if self.bot_playing:
                self._pending_start = True
                self._hold_until = now + INTERRUPT_MIN_S
                trace_stream(
                    "speech",
                    "vad.echo_hold",
                    hold_s=INTERRUPT_MIN_S,
                    rms=int(self.last_rms),
                )
                return None
            return self._emit_start(now)
        if kind in {"end_of_speech", "stop"}:
            self._pending_start = False
            self._hold_until = 0.0
            if not self.speaking:
                return None
            dur = (now - self._speech_start) if self._speech_start else -1.0
            self.speaking = False
            self._speech_start = None
            trace_stream(
                "speech",
                "vad.stop",
                rms=int(self.last_rms),
                utterance_s=round(dur, 2),
                bot_playing=self.bot_playing,
                backend=self.backend,
            )
            return "stop"
        if kind in {"inference_done", "inference"}:
            return self._maybe_release_hold(now)
        return None

    def _emit_start(self, now: float) -> str:
        self.speaking = True
        self._pending_start = False
        self._hold_until = 0.0
        self._speech_start = now
        trace_stream(
            "speech",
            "vad.start",
            rms=int(self.last_rms),
            bot_playing=self.bot_playing,
            backend=self.backend,
        )
        return "start"

    def _maybe_release_hold(self, now: float) -> str | None:
        if not self._pending_start or self.speaking:
            return None
        if self.bot_playing and now < self._hold_until:
            return None
        return self._emit_start(now)

    def feed(self, pcm16: bytes) -> str | None:
        rms = pcm_rms(pcm16)
        self.last_rms = rms
        self.frame_seq += 1
        now = time.monotonic()
        if hires_enabled():
            trace_stream(
                "speech",
                "vad.frame",
                frame_seq=self.frame_seq,
                bytes=len(pcm16),
                rms=rms,
                speaking=self.speaking,
                bot_playing=self.bot_playing,
                backend=self.backend,
            )
        self._stats.add(
            frames=1,
            rms=rms,
            speaking=int(self.speaking),
            bot_playing=int(self.bot_playing),
        )
        self._push_frame(pcm16)
        if self._edges:
            silero_edge = self._edges.popleft()
        else:
            silero_edge = self._maybe_release_hold(now)
        self._energy.bot_playing = self.bot_playing
        energy_edge = self._energy.feed(pcm16)
        if silero_edge:
            self._assist_active = False
            return silero_edge
        if energy_edge == "start" and not self.speaking:
            self._assist_active = True
            trace_stream(
                "speech",
                "vad.energy_assist",
                edge="start",
                rms=int(self.last_rms),
                backend=self.backend,
            )
            return self._emit_start(now)
        if energy_edge == "stop" and self._assist_active and self.speaking:
            self._assist_active = False
            trace_stream(
                "speech",
                "vad.energy_assist",
                edge="stop",
                rms=int(self.last_rms),
                backend=self.backend,
            )
            return self.apply_event("end_of_speech", now=now)
        return None

    def _push_frame(self, pcm16: bytes) -> None:
        stream = self._stream
        rtc = self._rtc
        if stream is None or rtc is None or not pcm16:
            return
        n = len(pcm16) // 2
        if n <= 0:
            return
        try:
            stream.push_frame(
                rtc.AudioFrame(
                    data=pcm16[: n * 2],
                    sample_rate=MODEL_RATE,
                    num_channels=1,
                    samples_per_channel=n,
                )
            )
        except Exception as exc:
            LOG.debug("silero push_frame failed: %s", exc)

    def _note_edge(self, edge: str | None) -> None:
        if edge:
            self._edges.append(edge)

    async def start(self) -> None:
        from livekit import rtc
        from livekit.agents.inference import VAD as InferenceVAD

        self._rtc = rtc
        threshold = float(os.environ.get("VAD_CONFIDENCE") or "0.5")
        # livekit-local-inference is CPU-only. Constructor already sets
        # VADCapabilities(update_interval=0.032) — one 32 ms window.
        self._vad = InferenceVAD(
            model="silero",
            min_speech_duration=self.start_secs,
            min_silence_duration=self.stop_secs,
            activation_threshold=threshold,
        )
        self._stream = self._vad.stream()
        self._task = asyncio.create_task(self._consume(), name="silero-vad")
        trace_stream(
            "speech",
            "vad.silero_start",
            min_speech_s=self.start_secs,
            min_silence_s=self.stop_secs,
            activation_threshold=threshold,
        )

    async def _consume(self) -> None:
        from livekit.agents import vad as lk_vad

        stream = self._stream
        if stream is None:
            return
        try:
            async for ev in stream:
                ev_type = getattr(ev.type, "value", None) or str(ev.type)
                if ev.type == lk_vad.VADEventType.START_OF_SPEECH:
                    ev_type = "start_of_speech"
                elif ev.type == lk_vad.VADEventType.END_OF_SPEECH:
                    ev_type = "end_of_speech"
                elif ev.type == lk_vad.VADEventType.INFERENCE_DONE:
                    ev_type = "inference_done"
                edge = self.apply_event(ev_type, probability=float(getattr(ev, "probability", 0.0) or 0.0))
                self._note_edge(edge)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOG.warning("silero VAD stream ended: %s", exc)

    async def close(self) -> None:
        task = self._task
        self._task = None
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.end_input()
            except Exception:
                pass
            closer = getattr(stream, "aclose", None)
            if callable(closer):
                try:
                    await closer()
                except Exception:
                    pass
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


def make_vad(
    *,
    start_secs: float | None = None,
    stop_secs: float | None = None,
) -> EnergyVad | LiveKitSileroVad:
    """Build the VAD selected by ``VAD_BACKEND`` (silero default, energy fallback)."""
    backend = (os.environ.get("VAD_BACKEND") or "silero").strip().lower()
    if backend in {"silero", "livekit"}:
        try:
            import livekit.agents.inference  # noqa: F401

            return LiveKitSileroVad(start_secs=start_secs, stop_secs=stop_secs)
        except Exception as exc:
            LOG.warning("Silero VAD unavailable (%s); using energy VAD", exc)
    return EnergyVad(start_secs=start_secs, stop_secs=stop_secs)
