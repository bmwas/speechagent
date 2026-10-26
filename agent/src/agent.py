"""Speech session: LiveKit bot + MiniCPM duplex or Qwen turn-based omni."""

from __future__ import annotations

import asyncio
import logging
import os
import time

from audio import ClockedPlayout, LinearResampler, MODEL_RATE
from barge_in import make_vad
from duplex_model import MiniCPMDuplexEngine
from livekit_room import LiveKitBotRoom, cancel_task_robust
from native_minicpm import NativeMiniCPMEngine
from qwen_omni import QwenOmniEngine
from session_score import SessionScore, TurnSample
from tokens import BOT_IDENTITY, token_for_bot
from trace import RateLimiter, StreamMeter, hires_enabled, set_session_id, set_trace_id, trace, trace_dbg, trace_exc, trace_stream
from video import FrameSampler
from voices import DEFAULT_VOICE_ID, asset_path, resolve_voice
from qwen_speakers import default_qwen_speaker_id, resolve_qwen_speaker

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
    # MiniCPM-o picks language from this string (official English clone
    # prompt). A generic assistant prompt plus HT_ref_audio.wav falls back
    # to Chinese after a thinker overflow / audio-encoder KV reset.
    return (
        os.environ.get("AGENT_SYSTEM_PROMPT")
        or "Clone the voice in the provided audio prompt. Please assist users "
        "while maintaining this voice style. Please answer the user's questions "
        "seriously and in a high quality. Please chat with the user in a highly "
        "human-like and oral style. You can see the user through their camera "
        "and hear them through their microphone. Always reply in English. Never "
        "switch to Chinese unless the user is speaking Chinese. Keep responses "
        "concise. You are a helpful assistant developed by ModelBest: MiniCPM-Omni."
    ).strip()


class SpeechSession:
    def __init__(self, room_id: str, on_stopped=None, *, voice_id: str = "") -> None:
        self.room_id = room_id
        self.session_id = f"lk-{room_id}"
        self._on_stopped = on_stopped
        self.model_name = os.environ.get("MODEL_NAME", "openbmb/MiniCPM-o-4_5")
        self.minicpm = is_minicpm(self.model_name)
        prompt = system_prompt()
        backend = (os.environ.get("MODEL_BACKEND") or "vllm").strip().lower()
        self.native = backend == "native"
        if self.native and not self.minicpm:
            raise RuntimeError(
                "MODEL_BACKEND=native requires MODEL_NAME containing 'minicpm'. "
                "Qwen is vLLM-only. Set MODEL_BACKEND=vllm or "
                "MODEL_NAME=openbmb/MiniCPM-o-4_5."
            )
        self.engine: MiniCPMDuplexEngine | QwenOmniEngine | NativeMiniCPMEngine
        if self.minicpm:
            # MiniCPM: validate against WAV catalog
            spec = resolve_voice(voice_id or DEFAULT_VOICE_ID)
            self.voice_id = spec.id
            ref = None if spec.id == DEFAULT_VOICE_ID else asset_path(spec)
            if self.native:
                self.engine = NativeMiniCPMEngine(prompt, voice_id=spec.id, ref_path=ref)
            else:
                self.engine = MiniCPMDuplexEngine(prompt, voice_id=spec.id, ref_path=ref)
        else:
            # Qwen: use built-in speaker (no WAV)
            qwen_spec = resolve_qwen_speaker(voice_id or default_qwen_speaker_id())
            self.voice_id = qwen_spec.id
            self.engine = QwenOmniEngine(prompt, speaker=qwen_spec.id)
        # MiniCPM (vLLM) commits the user turn on VAD stop. 0.80 s chops
        # mid-thought pauses into extra commits; the model then `listen`s
        # and the user has to repeat. Qwen still uses VAD_STOP_SECS.
        # Native duplex does not commit on stop (OpenBMB #17).
        vad_stop = (
            _env_float("MINICPM_VAD_STOP_SECS", 1.2)
            if self.minicpm
            else _env_float("VAD_STOP_SECS", 0.80)
        )
        self.vad = make_vad(stop_secs=vad_stop)
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
        self._mic_packet_seq = 0
        self._last_mic = 0.0
        self._gap_open = False
        self._engine_down_since = 0.0
        self._engine_down_rl = RateLimiter(10.0)
        self._pull_n = 0
        self._pull_bytes = 0
        self._pull_last_t = 0.0
        self._empty_polls = 0
        self._pull_chunk_rl = RateLimiter(2.0)
        self._pull_empty_rl = RateLimiter(0.1)
        self._prev_source_active: bool | None = None
        self._mic_meter = StreamMeter("speech", "stream.mic_in", kind="in")
        self._pull_meter = StreamMeter("speech", "stream.playout_pull", kind="out")
        # Session lab turn tracking
        self._session_score = SessionScore()
        self._last_engine_speaking = False
        self._t_session_start = time.monotonic()
        self._last_turn_sample: TurnSample | None = None

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
            path="native" if self.native else ("minicpm" if self.minicpm else "qwen"),
            voice_id=self.voice_id,
            vad_backend=getattr(self.vad, "backend", "energy"),
            vad_stop_secs=self.vad.stop_secs,
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
        start_vad = getattr(self.vad, "start", None)
        if callable(start_vad):
            await start_vad()
        self._play_task = asyncio.create_task(self._playback_loop())
        self._clock_task = asyncio.create_task(self._clock_loop())
        self._health_task = asyncio.create_task(self._health_loop())

    async def run_brain(self) -> None:
        if self._brain_started:
            return
        self._brain_started = True
        try:
            await self.engine.start()
            trace_stream(
                "speech",
                "session.brain_ready",
                model=self.model_name,
                path="native" if self.native else ("minicpm" if self.minicpm else "qwen"),
                voice_id=self.voice_id,
            )
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
        # Qwen emits two coarse vocoder blobs ~1.2 s apart. During that hole
        # audible=False and ring_ms=0 while generating is still true. Echo
        # hold and energy-assist must see occupying, or they barge the rest
        # of the sentence.
        generating = bool(getattr(self.engine, "generating", False))
        occupying = bool(
            self.playout.audible
            or generating
            or self.playout.ring_ms > 0
            or getattr(self.playout, "source_active", False)
        )
        self.vad.bot_playing = occupying
        edge = self.vad.feed(pcm16)
        self._mic_packet_seq += 1
        if hires_enabled():
            trace_stream(
                "speech",
                "stream.mic_packet",
                packet_seq=self._mic_packet_seq,
                bytes=len(pcm16),
                src_rate=int(rate) or 48000,
                dst_rate=MODEL_RATE,
                rms=self.vad.last_rms,
                edge=edge or "",
                is_speech=bool(self.vad.speaking or edge in ("start", "stop")),
                bot_playing=self.vad.bot_playing,
                generating=bool(getattr(self.engine, "generating", False)),
            )
        if edge == "start":
            generating = bool(getattr(self.engine, "generating", False))
            occupying = bool(
                self.playout.audible
                or generating
                or self.playout.ring_ms > 0
                or getattr(self.playout, "source_active", False)
            )
            trace_stream(
                "speech",
                "stream.user_utt_start",
                rms=self.vad.last_rms,
                identity=identity,
                bot_playing=self.vad.bot_playing,
                generating=generating,
                occupying=occupying,
                ring_ms=round(self.playout.ring_ms, 1),
                engine_ready=bool(getattr(self.engine, "ready", False)),
            )
            if occupying:
                await self.engine.interrupt()
                self.playout.clear()
                if self.bot is not None:
                    self.bot.clear_source_queue()
                trace_stream(
                    "speech",
                    "turn.barge_in",
                    source=getattr(self.vad, "backend", "energy"),
                    identity=identity,
                    bot_playing=self.vad.bot_playing,
                    generating=generating,
                    ring_ms=round(self.playout.ring_ms, 1),
                    during_stall=int(generating and not self.vad.bot_playing),
                )
        generating = bool(getattr(self.engine, "generating", False))
        if edge == "stop":
            trace_stream(
                "speech",
                "stream.user_utt_end",
                rms=self.vad.last_rms,
                identity=identity,
                generating=generating,
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
        # Silero can fire END_OF_SPEECH more often than energy VAD. Do not
        # re-commit / re-query while the model is still producing a reply.
        # Native duplex never commits on stop (OpenBMB #17).
        if edge == "stop" and not generating:
            if not self.minicpm:
                gen = getattr(self.engine, "generate", None)
                if callable(gen):
                    await gen()
            elif not self.native:
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
                # Tell the jitter buffer whether the model is still producing,
                # so it waits for the prebuffer while generating but flushes the
                # tail of a short reply immediately when generation stops.
                generating = bool(getattr(self.engine, "generating", False))
                self.playout.source_active = generating
                if generating != self._prev_source_active:
                    snap = {}
                    getter = getattr(self.engine, "stream_snapshot", None)
                    if callable(getter):
                        try:
                            snap = getter() or {}
                        except Exception:
                            snap = {}
                    trace_stream(
                        "speech",
                        "playout.source_active_change",
                        source_active=generating,
                        prev=self._prev_source_active,
                        ring_ms=round(self.playout.ring_ms, 1),
                        generating=generating,
                        turn_deltas=snap.get("turn_deltas", -1),
                    )
                    self._prev_source_active = generating
                    # Collect turn sample when model stops generating.
                    # Must not raise — this loop is the only audio path.
                    if not generating and self._last_engine_speaking:
                        try:
                            self._collect_turn_sample(snap)
                        except Exception:
                            LOG.exception("turn sample")
                self._last_engine_speaking = generating
                for pcm, rate in chunks:
                    self.playout.push(pcm, int(rate) or 24000)
                    pulled += len(pcm)
                if pulled:
                    now = time.monotonic()
                    pull_gap_ms = int((now - self._pull_last_t) * 1000) if self._pull_last_t else -1
                    self._pull_last_t = now
                    self._pull_n += 1
                    self._pull_bytes += pulled
                    ring_ms = self.playout.ring_ms
                    snap = {}
                    getter = getattr(self.engine, "stream_snapshot", None)
                    if callable(getter):
                        try:
                            snap = getter() or {}
                        except Exception:
                            snap = {}
                    self._pull_meter.add(
                        chunks=len(chunks),
                        bytes=pulled,
                        ring_ms=ring_ms,
                        pull_gap_ms=pull_gap_ms,
                        empty_polls=self._empty_polls,
                    )
                    if hires_enabled() or self._pull_n <= 5 or pull_gap_ms >= 250 or self._pull_chunk_rl.due():
                        trace_stream(
                            "speech",
                            "pull.chunk",
                            pull_gap_ms=pull_gap_ms,
                            empty_polls=self._empty_polls,
                            chunks=len(chunks),
                            bytes=pulled,
                            ring_ms=round(ring_ms, 1),
                            response_id=snap.get("response_id", ""),
                            turn_deltas=snap.get("turn_deltas", -1),
                        )
                    self._empty_polls = 0
                elif generating:
                    self._empty_polls += 1
                    if hires_enabled() and self._pull_empty_rl.due():
                        trace_stream(
                            "speech",
                            "pull.empty",
                            empty_polls=self._empty_polls,
                            ring_ms=round(self.playout.ring_ms, 1),
                        )
                self.vad.bot_playing = self.playout.audible
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
                        "path": "native" if self.native else ("minicpm" if self.minicpm else "qwen"),
                        "vad_backend": getattr(self.vad, "backend", "energy"),
                        "humans": len(self._humans),
                        "brain_started": self._brain_started,
                        "ready": ready,
                        "generating": generating,
                        "vad_speaking": self.vad.speaking,
                        "vad_rms": self.vad.last_rms,
                        "bot_playing": self.vad.bot_playing,
                        "playout_audible": self.playout.audible,
                        "ring_ms": round(self.playout.ring_ms, 1),
                        "underruns_total": getattr(self.playout, "underruns_total", 0),
                        "lk_mic_age_ms": mic_age_ms,
                        "mic_first": self._mic_first,
                        "pull_n": self._pull_n,
                        "pull_bytes": self._pull_bytes,
                        "voice_id": self.voice_id,
                    }
                )
                trace_stream("speech", "stream.health", **fields)
                if self._humans and self._brain_started and not ready:
                    if not self._engine_down_since:
                        self._engine_down_since = now
                    elif (now - self._engine_down_since) >= 10.0 and self._engine_down_rl.due():
                        trace_stream(
                            "speech",
                            "stream.engine_down",
                            down_ms=int((now - self._engine_down_since) * 1000),
                            path=fields.get("path", ""),
                            model=self.model_name,
                            ws_attempts=snap.get("ws_attempts", -1),
                            model_tcp_ok=snap.get("model_tcp_ok"),
                            model_probe_detail=snap.get("model_probe_detail", ""),
                        )
                else:
                    self._engine_down_since = 0.0
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

    def _collect_turn_sample(self, snap: dict) -> None:
        """Collect turn metrics from engine snapshot and playout stats."""
        playout_stats = self.playout.turn_stats()
        ttfa = snap.get("ttfa_ms", -1.0)
        ttft = snap.get("ttft_ms", -1.0)
        rtf = snap.get("rtf", -1.0)
        gap = snap.get("gap_ms", -1.0)
        audio_ms = snap.get("turn_audio_ms", 0.0)
        prebuffer = playout_stats.get("prebuffer_ms", -1.0)
        # E2E = TTFA + prebuffer (user-perceived latency to first audible)
        e2e = -1.0
        if ttfa >= 0 and prebuffer >= 0:
            e2e = ttfa + prebuffer
        sample = TurnSample(
            ttfa_ms=ttfa,
            ttfa_anchor=str(snap.get("ttfa_anchor", "")),
            ttft_ms=ttft,
            rtf=rtf,
            gap_ms=gap,
            audio_ms=audio_ms,
            prebuffer_ms=prebuffer,
            e2e_ms=e2e,
        )
        self._last_turn_sample = sample
        self._session_score.add_turn(sample)
        # Update session-level accumulators
        if audio_ms > 0:
            self._session_score.agent_audible_s += audio_ms / 1000.0
        trace_stream(
            "speech",
            "session.turn_sample",
            ttfa_ms=round(ttfa, 1),
            ttft_ms=round(ttft, 1),
            rtf=round(rtf, 2) if rtf >= 0 else -1,
            gap_ms=round(gap, 1) if gap >= 0 else -1,
            prebuffer_ms=round(prebuffer, 1) if prebuffer >= 0 else -1,
            e2e_ms=round(e2e, 1) if e2e >= 0 else -1,
            turns=len(self._session_score.turns),
        )

    def session_score_snapshot(self) -> dict:
        """Return session score snapshot for /api/live."""
        now = time.monotonic()
        duration_s = now - self._t_session_start
        snap = self._session_score.snapshot()
        snap["duration_s"] = round(duration_s, 1)
        snap["underruns_total"] = self.playout.underruns_total
        snap["clock_slips"] = self.playout.clock_slips
        return snap

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
        close_vad = getattr(self.vad, "close", None)
        if callable(close_vad):
            try:
                await close_vad()
            except Exception:
                LOG.exception("vad close")
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
