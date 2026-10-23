"""LiveKit rtc.Room wrapper: subscribe to user A/V, publish return audio."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any

from audio import WEBRTC_CHANNELS, WEBRTC_FRAME_BYTES, WEBRTC_FRAME_SAMPLES, WEBRTC_RATE, LinearResampler, downmix_to_mono
from tokens import BOT_IDENTITY, livekit_url
from trace import RateLimiter, StreamMeter, hires_enabled, trace_exc, trace_stream

LOG = logging.getLogger("livekit_room")
SOURCE_QUEUE_MS = int(os.environ.get("AUDIO_SOURCE_QUEUE_MS", "120"))


def _pub_meta(args: tuple[Any, ...]) -> tuple[str, str]:
    ident = ""
    kind = ""
    for arg in args:
        ident = ident or str(getattr(arg, "identity", "") or "")
        participant = getattr(arg, "participant", None)
        if participant is not None:
            ident = ident or str(getattr(participant, "identity", "") or "")
        kind_val = getattr(arg, "kind", None)
        if kind_val is not None:
            kind = kind or str(kind_val)
        source = getattr(arg, "source", None)
        if source is not None and not kind:
            kind = str(source)
    return ident, kind


async def cancel_task_robust(task: asyncio.Task, label: str = "", attempts: int = 6) -> None:
    name = label or getattr(task, "get_name", lambda: "?")()
    requested = False
    for attempt in range(max(1, attempts)):
        if task.done():
            break
        try:
            task.cancel()
            requested = True
            break
        except RecursionError:
            LOG.warning("cancel_task_robust %s recursion (attempt %d)", name, attempt + 1)
            await asyncio.sleep(0)
    if not requested and not task.done():
        return
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


def _pcm16_frame(rtc_mod: Any, pcm: bytes) -> Any:
    n = WEBRTC_FRAME_SAMPLES
    if len(pcm) < WEBRTC_FRAME_BYTES:
        pcm = pcm + b"\x00" * (WEBRTC_FRAME_BYTES - len(pcm))
    pcm = pcm[:WEBRTC_FRAME_BYTES]
    create = getattr(rtc_mod.AudioFrame, "create", None)
    if callable(create):
        frame = create(WEBRTC_RATE, WEBRTC_CHANNELS, n)
        dest = getattr(frame, "data", None)
        if dest is not None:
            try:
                view = memoryview(dest).cast("B")
                view[:WEBRTC_FRAME_BYTES] = pcm
                return frame
            except (TypeError, ValueError, AttributeError):
                pass
    return rtc_mod.AudioFrame(
        data=pcm,
        sample_rate=WEBRTC_RATE,
        num_channels=WEBRTC_CHANNELS,
        samples_per_channel=n,
    )


class LiveKitBotRoom:
    def __init__(
        self,
        token: str,
        *,
        on_audio: Callable[[bytes, int, str], Awaitable[None]] | None = None,
        on_video: Callable[[bytes, tuple[int, int], str, str], Awaitable[None]] | None = None,
        on_participant: Callable[[str, str], Awaitable[None]] | None = None,
        on_disconnect: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.token = token
        self.on_audio = on_audio
        self.on_video = on_video
        self.on_participant = on_participant
        self.on_disconnect = on_disconnect
        self.room: Any = None
        self.source: Any = None
        self._tasks: set[asyncio.Task] = set()
        self._closed = asyncio.Event()
        self._seen_tracks: set[str] = set()
        self._out_buf = bytearray()
        self._out_resampler: LinearResampler | None = None
        self._rtc: Any = None
        self._quality_rl = RateLimiter(5.0)
        self._audio_in_rl = RateLimiter(0.1)
        self._capture_rl = RateLimiter(0.1)
        self._pub_meter = StreamMeter("speech", "stream.livekit_out", kind="out")

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _bind_handlers(self) -> None:
        rtc = self._rtc
        room = self.room

        @room.on("participant_connected")
        def _joined(participant: rtc.RemoteParticipant) -> None:
            ident = getattr(participant, "identity", "") or ""
            if ident == BOT_IDENTITY:
                return
            trace_stream("speech", "livekit.participant_join", identity=ident)
            if self.on_participant:
                self._spawn(self.on_participant(ident, "joined"))

        @room.on("participant_disconnected")
        def _left(participant: rtc.RemoteParticipant) -> None:
            ident = getattr(participant, "identity", "") or ""
            if ident == BOT_IDENTITY:
                return
            trace_stream("speech", "livekit.participant_leave", identity=ident)
            if self.on_participant:
                self._spawn(self.on_participant(ident, "left"))

        @room.on("track_subscribed")
        def _sub(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            self._start_track(track, participant)

        @room.on("track_muted")
        def _muted(*args) -> None:
            try:
                ident, kind = _pub_meta(args)
                trace_stream("speech", "track.muted", identity=ident, kind=kind)
            except Exception:
                LOG.debug("track_muted handler", exc_info=True)

        @room.on("track_unmuted")
        def _unmuted(*args) -> None:
            try:
                ident, kind = _pub_meta(args)
                trace_stream("speech", "track.unmuted", identity=ident, kind=kind)
            except Exception:
                LOG.debug("track_unmuted handler", exc_info=True)

        @room.on("connection_quality_changed")
        def _quality(*args) -> None:
            if not self._quality_rl.due():
                return
            try:
                ident = ""
                quality = ""
                for arg in args:
                    ident = ident or str(getattr(arg, "identity", "") or "")
                    quality = quality or str(arg)
                trace_stream("speech", "livekit.quality", identity=ident, quality=quality[:80])
            except Exception:
                LOG.debug("connection_quality handler", exc_info=True)

        @room.on("reconnected")
        def _reconnected(*_a) -> None:
            trace_stream("speech", "livekit.reconnect")

        @room.on("disconnected")
        def _disc(*_a):
            trace_stream("speech", "livekit.disconnect")
            if self._closed.is_set():
                return
            self._closed.set()
            if self.on_disconnect:
                self._spawn(self.on_disconnect())

    def _track_key(self, track: Any, identity: str) -> str:
        sid = getattr(track, "sid", None) or getattr(track, "name", None)
        return f"{identity}:{sid or id(track)}"

    def _start_track(self, track: Any, participant: Any) -> None:
        rtc = self._rtc
        ident = getattr(participant, "identity", "") or ""
        if ident == BOT_IDENTITY:
            return
        key = self._track_key(track, ident)
        if key in self._seen_tracks:
            return
        self._seen_tracks.add(key)
        kind = getattr(track, "kind", None)
        if kind == rtc.TrackKind.KIND_AUDIO:
            trace_stream("speech", "track.audio_sub", identity=ident)
            self._spawn(self._audio_loop(rtc.AudioStream(track), ident))
        elif kind == rtc.TrackKind.KIND_VIDEO:
            trace_stream("speech", "track.video_sub", identity=ident)
            stream = rtc.VideoStream(track, format=rtc.VideoBufferType.RGB24)
            self._spawn(self._video_loop(stream, ident))

    def _subscribe_existing(self) -> None:
        room = self.room
        if room is None:
            return
        remotes = getattr(room, "remote_participants", None) or {}
        participants = list(remotes.values()) if isinstance(remotes, dict) else list(remotes)
        for participant in participants:
            ident = getattr(participant, "identity", "") or ""
            if ident == BOT_IDENTITY:
                continue
            if self.on_participant:
                self._spawn(self.on_participant(ident, "joined"))
            pubs = getattr(participant, "track_publications", None) or {}
            publications = list(pubs.values()) if isinstance(pubs, dict) else list(pubs)
            for pub in publications:
                track = getattr(pub, "track", None)
                if track is not None:
                    self._start_track(track, participant)

    async def connect(self) -> Any:
        from livekit import rtc

        self._rtc = rtc
        self.room = rtc.Room()
        url = livekit_url()
        self._bind_handlers()
        await self.room.connect(url, self.token)
        trace_stream("speech", "livekit.connect", url=url, role="bot")
        self._subscribe_existing()
        self.source = rtc.AudioSource(WEBRTC_RATE, WEBRTC_CHANNELS, SOURCE_QUEUE_MS)
        track = rtc.LocalAudioTrack.create_audio_track("assistant", self.source)
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE
        await self.room.local_participant.publish_track(track, options)
        trace_stream("speech", "track.audio_pub", queue_ms=SOURCE_QUEUE_MS)
        return self.source

    async def _audio_loop(self, stream, identity: str) -> None:
        try:
            async for event in stream:
                frame = event.frame
                pcm = bytes(frame.data)
                rate = int(getattr(frame, "sample_rate", 0) or WEBRTC_RATE)
                channels = int(getattr(frame, "num_channels", 1) or 1)
                if channels >= 2:
                    pcm = downmix_to_mono(pcm, channels)
                if hires_enabled() and self._audio_in_rl.due():
                    trace_stream(
                        "speech",
                        "livekit.audio_in",
                        bytes=len(pcm),
                        rate=rate,
                        channels=channels,
                        identity=identity,
                    )
                if self.on_audio:
                    await self.on_audio(pcm, rate, identity)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            trace_exc("speech", "error.audio_loop", exc, identity=identity)

    async def _video_loop(self, stream, identity: str) -> None:
        try:
            async for event in stream:
                buf = event.frame
                w = int(getattr(buf, "width", 0) or 0)
                h = int(getattr(buf, "height", 0) or 0)
                raw = bytes(buf.data)
                if w <= 0 or h <= 0 or not raw:
                    continue
                if self.on_video:
                    await self.on_video(raw, (w, h), "RGB", identity)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            trace_exc("speech", "error.video_loop", exc, identity=identity)

    async def capture_pcm16(self, pcm: bytes, sample_rate: int) -> None:
        from livekit import rtc

        if self._closed.is_set() or self.source is None or not pcm:
            return
        rate = int(sample_rate) or WEBRTC_RATE
        if rate != WEBRTC_RATE:
            if self._out_resampler is None or self._out_resampler.dst_rate != WEBRTC_RATE:
                self._out_resampler = LinearResampler(WEBRTC_RATE)
            pcm = self._out_resampler.feed(pcm, rate)
            if not pcm:
                return
        self._out_buf.extend(pcm)
        while len(self._out_buf) >= WEBRTC_FRAME_BYTES:
            chunk = bytes(self._out_buf[:WEBRTC_FRAME_BYTES])
            del self._out_buf[:WEBRTC_FRAME_BYTES]
            frame = _pcm16_frame(rtc, chunk)
            t0 = time.monotonic()
            try:
                await self.source.capture_frame(frame)
            except RuntimeError:
                return
            block_ms = (time.monotonic() - t0) * 1000.0
            self._pub_meter.add(frames=1, bytes=len(chunk), block_ms=block_ms)
            if hires_enabled() and self._capture_rl.due():
                trace_stream(
                    "speech",
                    "livekit.capture",
                    bytes=len(chunk),
                    block_ms=round(block_ms, 1),
                    out_buf_bytes=len(self._out_buf),
                )
            if block_ms >= 50:
                trace_stream("speech", "playout.block", block_ms=int(block_ms), where="livekit_capture")

    def clear_source_queue(self) -> None:
        clear = getattr(self.source, "clear_queue", None)
        if callable(clear):
            try:
                clear()
                trace_stream("speech", "turn.source_queue_clear", source_queue_ms=SOURCE_QUEUE_MS)
            except Exception:
                LOG.exception("clear_source_queue")

    async def close(self) -> None:
        try:
            self._pub_meter.flush()
        except Exception:
            pass
        self._closed.set()
        for task in list(self._tasks):
            await cancel_task_robust(task, label="livekit_room")
        self._tasks.clear()
        self._seen_tracks.clear()
        self._out_buf.clear()
        if self.room is not None:
            try:
                await self.room.disconnect()
            except Exception:
                LOG.exception("room disconnect")
            self.room = None
