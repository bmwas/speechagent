"""Energy VAD + echo guard. MiniCPM-o 4.5 does not self-yield mid-turn."""

from __future__ import annotations

import logging
import os
import time

from audio import pcm_rms
from trace import STREAM_PERIOD_S, StreamMeter, hires_enabled, trace_stream

LOG = logging.getLogger("barge_in")

VAD_START_SECS = float(os.environ.get("VAD_START_SECS", "0.15"))
VAD_STOP_SECS = float(os.environ.get("VAD_STOP_SECS", "0.80"))
MIC_GATE_RMS = float(os.environ.get("VAD_MIN_VOLUME", "0.01")) * 32768.0
MIC_GATE_MIN = 80.0
MIC_GATE_MULT = 3.5
INTERRUPT_MIN_S = 0.50


class EnergyVad:
    """RMS hangover VAD. Returns 'start' / 'stop' on edges, else None."""

    def __init__(self) -> None:
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
        if hires_enabled():
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
            if not self.speaking and (now - self._loud_since) >= VAD_START_SECS:
                if self.bot_playing:
                    held = now - self._loud_since
                    if held < INTERRUPT_MIN_S:
                        return None
                self.speaking = True
                self._speech_start = now
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
        if (now - self._quiet_since) >= VAD_STOP_SECS:
            dur = (now - self._speech_start) if self._speech_start else -1.0
            self.speaking = False
            self._speech_start = None
            self._quiet_since = None
            trace_stream(
                "speech",
                "vad.stop",
                rms=int(rms),
                utterance_s=round(dur, 2),
                bot_playing=self.bot_playing,
            )
            return "stop"
        return None
