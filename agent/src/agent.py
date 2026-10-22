"""Speech session: LiveKit bot + MiniCPM duplex or Qwen turn-based omni."""

from __future__ import annotations

import asyncio
import logging
import os
import time

from audio import ClockedPlayout, LinearResampler, MODEL_RATE
from barge_in import EnergyVad
from duplex_model import MiniCPMDuplexEngine
from livekit_room import LiveKitBotRoom, cancel_task_robust
from qwen_omni import QwenOmniEngine
from tokens import BOT_IDENTITY, token_for_bot
from trace import StreamMeter, set_session_id, set_trace_id, trace, trace_dbg, trace_exc, trace_stream
from video import FrameSampler

LOG = logging.getLogger("agent")


def is_minicpm(model: str | None = None) -> bool:
    name = (model or os.environ.get("MODEL_NAME") or "").lower()
    return "minicpm" in name


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def system_prompt() -> str:
    return (
        os.environ.get("AGENT_SYSTEM_PROMPT")
        or "You are a helpful, friendly AI assistant. You can see the user through "
        "their camera and hear them through their microphone. Respond naturally "
        "and conversationally. Keep responses concise and engaging."
    ).strip()


class SpeechSession:
    def __init__(self, room_id: str, on_stopped=None) -> None:
        self.room_id = room_id
        self.session_id = f"lk-{room_id}"
        self._on_stopped = on_stopped
        self.model_name = os.environ.get("MODEL_NAME", "openbmb/MiniCPM-o-4_5")
        self.minicpm = is_minicpm(self.model_name)
        prompt = system_prompt()
        self.engine: MiniCPMDuplexEngine | QwenOmniEngine
        if self.minicpm:
            self.engine = MiniCPMDuplexEngine(prompt)
        else:
            self.engine = QwenOmniEngine(prompt)
        self.vad = EnergyVad()
        self.playout = ClockedPlayout()
        if self.minicpm:
            self.frames = FrameSampler()
        else:
            self.frames = FrameSampler(
                fps=_env_float("QWEN_VIDEO_FPS", 5.0),
                max_side=_env_int("QWEN_VIDEO_MAX_SIDE", 448),
            )
        self.bot: LiveKitBotRoom | None = None
        self._in_rs = LinearResampler(MODEL_RATE)
        self._running = False
        self._brain_started = False
        self._play_task: asyncio.Task | None = None
        self._clock_task: asyncio.Task | None = None
        self._brain_task: asyncio.Task | None = None
        self._health_task: asyncio.Task | None = None
        self._humans: set[str] = set()
        self._mic_first = False
        self._last_mic = 0.0
        self._gap_open = False
        self._pull_n = 0
        self._pull_bytes = 0
        self._mic_meter = StreamMeter("speech", "stream.mic_in", kind="in")
        self._pull_meter = StreamMeter("speech", "stream.playout_pull", kind="out")

    async def connect_and_publish(self) -> None:
        if self._running:
            return
        self._running = True
        set_session_id(self.session_id)
        set_trace_id()
        trace(
            "speech",
            "session.open",
            room=self.room_id,
            model=self.model_name,
            path="minicpm" if self.minicpm else "qwen",
        )
        token = token_for_bot(self.room_id)
        self.bot = LiveKitBotRoom(
            token,
            on_audio=self._on_audio,
            on_video=self._on_video,
            on_participant=self._on_participant,
            on_disconnect=self.stop,
        )
        await self.bot.connect()
        self._play_task = asyncio.create_task(self._playback_loop())
        self._clock_task = asyncio.create_task(self._clock_loop())
        self._health_task = asyncio.create_task(self._health_loop())

    async def run_brain(self) -> None:
        if self._brain_started:
            return
        self._brain_started = True
        try:
            await self.engine.start()
            trace_stream("speech", "session.brain_ready", model=self.model_name, path="minicpm" if self.minicpm else "qwen")
        except Exception as exc:
            trace_exc("speech", "error.brain", exc)
            await self.stop()

    def spawn_brain(self) -> None:
        if self._brain_task is None or self._brain_task.done():
            self._brain_task = asyncio.create_task(self.run_brain())

    async def _on_participant(self, identity: str, action: str) -> None:
        if identity == BOT_IDENTITY:
            return
        if action == "joined":
            self._humans.add(identity)
        elif action == "left":
            self._humans.discard(identity)
            if not self._humans:
                trace("speech", "session.close", reason="empty_room")
                await self.stop()

    async def _on_audio(self, pcm: bytes, rate: int, identity: str) -> None:
        if identity == BOT_IDENTITY or not pcm:
            return
        pcm16 = self._in_rs.feed(pcm, int(rate) or 48000)
        if not pcm16:
            return
        self._last_mic = time.monotonic()
        if not self._mic_first:
            self._mic_first = True
            trace_stream(
                "speech",
                "stream.mic_first",
                bytes=len(pcm16),
                src_rate=int(rate) or 48000,
                identity=identity,
                brain_ready=self._brain_started and bool(getattr(self.engine, "ready", True)),
            )
        self.vad.bot_playing = bool(getattr(self.engine, "generating", False))
        edge = self.vad.feed(pcm16)
        if edge == "start":
            trace_stream(
                "speech",
                "stream.user_utt_start",
                rms=self.vad.last_rms,
                identity=identity,
                bot_playing=self.vad.bot_playing,
                engine_ready=bool(getattr(self.engine, "ready", False)),
            )
        if edge == "start" and self.vad.bot_playing:
            await self.engine.interrupt()
            self.playout.clear()
            if self.bot is not None:
                self.bot.clear_source_queue()
            trace_stream("speech", "turn.barge_in", source="energy", identity=identity)
        if edge == "stop":
            trace_stream(
                "speech",
                "stream.user_utt_end",
                rms=self.vad.last_rms,
                identity=identity,
                generating=bool(getattr(self.engine, "generating", False)),
                ring_ms=round(self.playout.ring_ms, 1),
            )
        is_speech = bool(self.vad.speaking or edge in ("start", "stop"))
        ready = bool(getattr(self.engine, "ready", True))
        self._mic_meter.add(
            frames=1,
            bytes=len(pcm16),
            rms=self.vad.last_rms,
            is_speech=int(is_speech),
            ready=int(ready),
            src_rate=int(rate) or 48000,
            identity=identity,
        )
        await self.engine.push_audio(pcm16, MODEL_RATE, is_speech=is_speech)
        if edge == "stop" and not self.minicpm:
            gen = getattr(self.engine, "generate", None)
            if callable(gen):
                await gen()
        if edge == "stop" and self.minicpm:
            end = getattr(self.engine, "end_of_speech", None)
            if callable(end):
                await end()
        if edge:
            trace_dbg("speech", "audio.chunk", rms=self.vad.last_rms, edge=edge, bytes=len(pcm16), is_speech=is_speech)

    async def _on_video(self, raw: bytes, size: tuple[int, int], fmt: str, identity: str) -> None:
        jpeg = self.frames.offer(raw, size, fmt)
        if jpeg:
            await self.engine.push_image(jpeg)

    async def _playback_loop(self) -> None:
        try:
            while self._running:
                chunks = await self.engine.pull_audio()
                pulled = 0
                for pcm, rate in chunks:
                    self.vad.bot_playing = True
                    self.playout.push(pcm, int(rate) or 24000)
                    pulled += len(pcm)
                if pulled:
                    self._pull_n += 1
                    self._pull_bytes += pulled
                    self._pull_meter.add(chunks=len(chunks), bytes=pulled, ring_ms=self.playout.ring_ms)
                if not chunks and not getattr(self.engine, "generating", False):
                    self.vad.bot_playing = False
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            trace_exc("speech", "error.playback", exc)

    async def _health_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(1.0)
                if not self._running:
                    break
                now = time.monotonic()
                mic_age_ms = int((now - self._last_mic) * 1000) if self._last_mic else -1
                snap = {}
                getter = getattr(self.engine, "stream_snapshot", None)
                if callable(getter):
                    try:
                        snap = getter() or {}
                    except Exception:
                        snap = {}
                ready = bool(getattr(self.engine, "ready", False))
                generating = bool(getattr(self.engine, "generating", False))
                fields = dict(snap)
                fields.update(
                    {
                        "path": "minicpm" if self.minicpm else "qwen",
                        "humans": len(self._humans),
                        "brain_started": self._brain_started,
                        "ready": ready,
                        "generating": generating,
                        "vad_speaking": self.vad.speaking,
                        "vad_rms": self.vad.last_rms,
                        "bot_playing": self.vad.bot_playing,
                        "ring_ms": round(self.playout.ring_ms, 1),
                        "lk_mic_age_ms": mic_age_ms,
                        "mic_first": self._mic_first,
                        "pull_n": self._pull_n,
                        "pull_bytes": self._pull_bytes,
                    }
                )
                trace_stream("speech", "stream.health", **fields)
                if self._humans and self._mic_first and mic_age_ms >= 1500:
                    if not self._gap_open:
                        self._gap_open = True
                        trace_stream(
                            "speech",
                            "stream.mic_gap",
                            gap_ms=mic_age_ms,
                            ready=ready,
                            generating=generating,
                        )
                elif self._gap_open:
                    self._gap_open = False
                    trace_stream("speech", "stream.mic_resume", gap_ms=mic_age_ms)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            trace_exc("speech", "error.stream_health", exc)

    async def _clock_loop(self) -> None:
        try:
            await self.playout.run(self._capture)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            trace_exc("speech", "error.playout_clock", exc)

    async def _capture(self, pcm: bytes, rate: int) -> None:
        if self.bot is not None:
            await self.bot.capture_pcm16(pcm, rate)

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self.playout.stop()
        current = asyncio.current_task()
        for meter in (self._mic_meter, self._pull_meter):
            try:
                meter.flush()
            except Exception:
                pass
        for attr in ("_brain_task", "_play_task", "_clock_task", "_health_task"):
            task = getattr(self, attr)
            if task is None or task is current or task.done():
                setattr(self, attr, None)
                continue
            await cancel_task_robust(task, label=attr)
            setattr(self, attr, None)
        try:
            await self.engine.close()
        except Exception:
            LOG.exception("engine close")
        if self.bot is not None:
            try:
                await self.bot.close()
            except Exception:
                LOG.exception("bot close")
            self.bot = None
        trace("speech", "session.close", room=self.room_id)
        cb = self._on_stopped
        self._on_stopped = None
        if cb is not None:
            try:
                cb()
            except Exception:
                LOG.exception("on_stopped")


def main() -> None:
    import sys

    from logging_setup import configure_logging
    from token_server import run

    _rlimit = int(os.environ.get("AGENT_RECURSION_LIMIT", "10000"))
    if sys.getrecursionlimit() < _rlimit:
        sys.setrecursionlimit(_rlimit)
    configure_logging()
    run()


if __name__ == "__main__":
    main()
