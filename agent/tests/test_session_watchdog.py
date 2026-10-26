"""Client-only turn watchdog: recover dead duplex sessions without protocol changes."""

from __future__ import annotations

import array
import asyncio
import json
import os
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from session_watchdog import TurnWatchdog  # noqa: E402
from duplex_model import MiniCPMDuplexEngine  # noqa: E402
from barge_in import EnergyVad, LiveKitSileroVad, VAD_START_SECS, VAD_STOP_SECS  # noqa: E402
from native_minicpm import (  # noqa: E402
    LISTEN_LOCK_N,
    SPEAK_BARGE_N,
    SPEAK_YIELD_N,
    NativeMiniCPMEngine,
    _f32le_to_pcm16,
    _pcm16_to_f32le,
    _worker_ws_url,
)


class TurnWatchdogTests(unittest.TestCase):
    def test_timeout_only_while_armed(self) -> None:
        dog = TurnWatchdog(timeout_ms=1000)
        now = 100.0
        self.assertEqual(dog.poll(now), "")
        dog.arm(now)
        self.assertEqual(dog.poll(now + 0.5), "")
        reason = dog.poll(now + 1.2)
        self.assertTrue(reason.startswith("decision_timeout_"))
        self.assertFalse(dog.armed)
        self.assertEqual(dog.poll(now + 5.0), "")

    def test_listen_is_a_valid_decision(self) -> None:
        dog = TurnWatchdog(timeout_ms=1000)
        now = 10.0
        dog.arm(now)
        self.assertTrue(dog.clear(now + 0.2, "listen"))
        self.assertEqual(dog.decision, "listen")
        self.assertEqual(dog.poll(now + 5.0), "")
        snap = dog.snapshot(now + 0.3)
        self.assertFalse(snap["turn_waiting"])
        self.assertEqual(snap["commit_age_ms"], 300)
        self.assertEqual(snap["decision_age_ms"], 100)

    def test_speak_and_first_audio_clear(self) -> None:
        for decision in ("created", "speak", "first_audio", "error", "cancel"):
            dog = TurnWatchdog(timeout_ms=8000)
            dog.arm(1.0)
            self.assertTrue(dog.clear(1.4, decision), decision)
            self.assertEqual(dog.poll(20.0), "")

    def test_session_generation_increments(self) -> None:
        dog = TurnWatchdog()
        self.assertEqual(dog.note_session(), 1)
        dog.arm(1.0)
        self.assertEqual(dog.note_session(), 2)
        self.assertFalse(dog.armed)
        self.assertEqual(dog.t_commit, 0.0)


class DuplexWatchdogTests(unittest.IsolatedAsyncioTestCase):
    async def test_snapshot_exposes_liveness(self) -> None:
        eng = MiniCPMDuplexEngine()
        snap = eng.stream_snapshot()
        self.assertIn("turn_waiting", snap)
        self.assertIn("commit_age_ms", snap)
        self.assertIn("decision_age_ms", snap)
        self.assertIn("session_generation", snap)
        self.assertIn("reconnect_reason", snap)
        self.assertFalse(snap["turn_waiting"])
        self.assertEqual(snap["session_generation"], 0)

    async def test_timeout_marks_unready_and_reconnects(self) -> None:
        eng = MiniCPMDuplexEngine()
        eng._ready.set()
        eng._turn.note_session()
        eng._turn.arm(time.monotonic() - 9.0)
        reconnects: list[str] = []

        async def fake_reconnect(reason: str) -> None:
            reconnects.append(reason)
            eng._ready.set()
            eng._turn.note_session()

        eng._reconnect = fake_reconnect  # type: ignore[method-assign]
        task = asyncio.create_task(eng._watchdog_loop())
        try:
            for _ in range(20):
                if reconnects:
                    break
                await asyncio.sleep(0.1)
            self.assertTrue(reconnects, "watchdog did not request reconnect")
            self.assertTrue(reconnects[0].startswith("decision_timeout_"))
            self.assertEqual(eng._last_reconnect_reason[:16], "decision_timeout")
            self.assertGreaterEqual(eng._turn.generation, 2)
        finally:
            eng._closing = True
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def test_listen_does_not_reconnect(self) -> None:
        eng = MiniCPMDuplexEngine()
        eng._ready.set()
        eng._turn.arm(time.monotonic() - 0.05)
        await eng._handle({"type": "response.listen"})
        self.assertFalse(eng._turn.armed)
        self.assertEqual(eng._turn.decision, "listen")
        self.assertEqual(eng._reconnect_reason, "")
        self.assertTrue(eng._ready.is_set())

    async def test_commit_always_arms_even_while_speaking(self) -> None:
        eng = MiniCPMDuplexEngine()
        sent: list[dict] = []

        async def fake_send(body: dict) -> None:
            sent.append(dict(body))

        eng._ws = object()
        eng._ready.set()
        eng._speaking = True
        eng._commit_pending = True
        eng._send = fake_send  # type: ignore[method-assign]
        await eng._commit_input()
        self.assertEqual(sent[0]["type"], "input_audio_buffer.commit")
        self.assertTrue(eng._turn.armed)

    async def test_speaking_flag_does_not_cancel_timeout(self) -> None:
        eng = MiniCPMDuplexEngine()
        eng._ready.set()
        eng._speaking = True
        eng._turn.note_session()
        eng._turn.arm(time.monotonic() - 9.0)
        reconnects: list[str] = []

        async def fake_reconnect(reason: str) -> None:
            reconnects.append(reason)
            eng._ready.set()
            eng._turn.note_session()

        eng._reconnect = fake_reconnect  # type: ignore[method-assign]
        task = asyncio.create_task(eng._watchdog_loop())
        try:
            for _ in range(20):
                if reconnects:
                    break
                await asyncio.sleep(0.1)
            self.assertTrue(reconnects, "watchdog did not request reconnect")
            self.assertTrue(reconnects[0].startswith("decision_timeout_"))
        finally:
            eng._closing = True
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def test_audio_delta_clears_armed_watchdog(self) -> None:
        eng = MiniCPMDuplexEngine()
        eng._turn.arm(time.monotonic())
        eng._speaking = True
        eng._note_decision("audio")
        self.assertFalse(eng._turn.armed)
        self.assertEqual(eng._turn.decision, "audio")

    async def test_commit_body_has_no_protocol_extras(self) -> None:
        eng = MiniCPMDuplexEngine()
        sent: list[dict] = []

        async def fake_send(body: dict) -> None:
            sent.append(dict(body))

        eng._ws = object()
        eng._ready.set()
        eng._commit_pending = True
        eng._send = fake_send  # type: ignore[method-assign]
        await eng._commit_input()
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["type"], "input_audio_buffer.commit")
        self.assertNotIn("response_required", sent[0])
        self.assertNotIn("require_response", sent[0])
        self.assertTrue(eng._turn.armed)

    async def test_commit_skipped_while_awaiting_decision(self) -> None:
        eng = MiniCPMDuplexEngine()
        sent: list[dict] = []

        async def fake_send(body: dict) -> None:
            sent.append(dict(body))

        eng._ws = object()
        eng._ready.set()
        eng._send = fake_send  # type: ignore[method-assign]
        eng._commit_pending = True
        await eng._commit_input()
        self.assertEqual(sum(1 for s in sent if s["type"] == "input_audio_buffer.commit"), 1)
        self.assertTrue(eng._turn.armed)
        eng._commit_pending = True
        await eng._commit_input()
        self.assertEqual(sum(1 for s in sent if s["type"] == "input_audio_buffer.commit"), 1)
        self.assertTrue(eng._commit_pending)

    async def test_listen_flushes_skipped_commit(self) -> None:
        eng = MiniCPMDuplexEngine()
        sent: list[dict] = []

        async def fake_send(body: dict) -> None:
            sent.append(dict(body))

        eng._ws = object()
        eng._ready.set()
        eng._send = fake_send  # type: ignore[method-assign]
        eng._commit_pending = True
        eng._last_is_speech = False
        await eng._commit_input()
        eng._commit_pending = True
        await eng._commit_input()
        self.assertEqual(sum(1 for s in sent if s["type"] == "input_audio_buffer.commit"), 1)
        await eng._handle({"type": "response.listen"})
        await eng._handle({"type": "response.listen"})
        self.assertIsNotNone(eng._listen_flush_task)
        await asyncio.sleep(0.25)
        commits = [s for s in sent if s["type"] == "input_audio_buffer.commit"]
        self.assertEqual(len(commits), 2)
        self.assertTrue(eng._turn.armed)
        self.assertFalse(eng._commit_pending)

    async def test_interrupt_cancels_and_drops_late_audio(self) -> None:
        import base64

        eng = MiniCPMDuplexEngine()
        sent: list[dict] = []

        async def fake_send(body: dict) -> None:
            sent.append(dict(body))

        eng._ws = object()
        eng._ready.set()
        eng._send = fake_send  # type: ignore[method-assign]
        eng._speaking = True
        eng._response_id = "resp-old"
        await eng._out.put((b"\x00\x00", 24000))
        await eng.interrupt()
        self.assertTrue(any(s.get("type") == "response.cancel" for s in sent))
        self.assertFalse(eng._speaking)
        self.assertTrue(eng._drop_output)
        self.assertTrue(eng._barge_until_commit)
        self.assertEqual(eng._out.qsize(), 0)
        pcm = b"\x00\x01" * 80
        b64 = base64.b64encode(pcm).decode("ascii")
        await eng._on_audio_delta(
            {"type": "response.audio.delta", "delta": b64, "response_id": "resp-old"}
        )
        self.assertEqual(eng._out.qsize(), 0)
        # auto_response must not play a new reply over the interrupt.
        await eng._handle({"type": "response.speak", "response_id": "resp-sneak"})
        self.assertTrue(eng._drop_output)
        await eng._on_audio_delta(
            {"type": "response.audio.delta", "delta": b64, "response_id": "resp-sneak"}
        )
        self.assertEqual(eng._out.qsize(), 0)
        eng._commit_pending = True
        await eng._commit_input()
        self.assertFalse(eng._barge_until_commit)
        await eng._handle({"type": "response.speak", "response_id": "resp-new"})
        self.assertFalse(eng._drop_output)
        await eng._on_audio_delta(
            {"type": "response.audio.delta", "delta": b64, "response_id": "resp-new"}
        )
        self.assertEqual(eng._out.qsize(), 1)

    async def test_hidden_state_error_reconnects(self) -> None:
        eng = MiniCPMDuplexEngine()
        eng._ready.set()
        eng._turn.arm(time.monotonic())
        await eng._handle(
            {
                "type": "error",
                "error": {"message": "MiniCPM-o native duplex: missing own-token hidden states for [151717]"},
            }
        )
        self.assertFalse(eng._ready.is_set())
        self.assertEqual(eng._reconnect_reason, "hidden_state_crash")
        self.assertFalse(eng._turn.armed)


class BaselineProtocolTests(unittest.TestCase):
    def test_no_require_response_or_token_patch(self) -> None:
        root = Path(__file__).resolve().parents[2]
        forbidden = (
            "response_required",
            "MINICPM_REQUIRE_RESPONSE",
            "listen_overridden",
            "_finalize_minicpmo45_native_duplex_sample",
        )
        scan = []
        for path in (
            root / "agent" / "src",
            root / "model-server",
            root / "docker-compose.yml",
        ):
            if path.is_file():
                scan.append(path)
            else:
                scan.extend(p for p in path.rglob("*") if p.is_file())
        hits: list[str] = []
        for path in scan:
            if path.suffix not in {".py", ".yml", ".sh", ".json", ".md"} and path.name != "docker-compose.yml":
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden:
                if token in text:
                    hits.append(f"{path}:{token}")
        self.assertEqual(hits, [])
        compose = (root / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertNotIn("/patches/apply.py", compose)
        self.assertNotIn("model-server/patches", compose)
        entry = (root / "model-server" / "entrypoint.sh").read_text(encoding="utf-8")
        self.assertNotIn("apply.py", entry)
        self.assertIn("hift-warmup-even-frames", entry)
        self.assertIn("hift-lazy-graphs-32", entry)


class EnergyVadTests(unittest.TestCase):
    def test_minicpm_stop_secs_override(self) -> None:
        vad = EnergyVad(stop_secs=1.2)
        self.assertEqual(vad.stop_secs, 1.2)
        self.assertEqual(vad.start_secs, VAD_START_SECS)

    def test_qwen_default_stop_secs(self) -> None:
        vad = EnergyVad()
        self.assertEqual(vad.stop_secs, VAD_STOP_SECS)

    def test_interrupt_hold_only_when_bot_playing(self) -> None:
        import array

        import barge_in as mod

        loud = array.array("h", [8000] * 1600).tobytes()
        clock = [0.0]
        orig = mod.time.monotonic
        mod.time.monotonic = lambda: clock[0]
        try:
            vad = EnergyVad(start_secs=0.15)
            self.assertIsNone(vad.feed(loud))
            clock[0] = 0.16
            self.assertEqual(vad.feed(loud), "start")

            vad = EnergyVad(start_secs=0.15)
            vad.bot_playing = True
            clock[0] = 0.0
            self.assertIsNone(vad.feed(loud))
            clock[0] = 0.20
            self.assertIsNone(vad.feed(loud))
            clock[0] = 0.51
            self.assertEqual(vad.feed(loud), "start")
        finally:
            mod.time.monotonic = orig

    def test_agent_barges_on_generating_not_only_audible(self) -> None:
        src = (Path(__file__).resolve().parents[1] / "src" / "agent.py").read_text(encoding="utf-8")
        self.assertIn("occupying", src)
        self.assertIn("during_stall", src)
        self.assertNotIn("turn.speech_during_model_stall", src)
        self.assertIn("not self.native", src)
        self.assertIn("not generating", src)


class SileroVadEventTests(unittest.TestCase):
    def test_start_without_bot_playing(self) -> None:
        vad = LiveKitSileroVad(start_secs=0.15, stop_secs=1.2)
        self.assertEqual(vad.apply_event("start_of_speech"), "start")
        self.assertTrue(vad.speaking)
        self.assertEqual(vad.apply_event("end_of_speech"), "stop")
        self.assertFalse(vad.speaking)

    def test_echo_hold_then_release(self) -> None:
        vad = LiveKitSileroVad()
        vad.bot_playing = True
        clock = 10.0
        self.assertIsNone(vad.apply_event("start_of_speech", now=clock))
        self.assertFalse(vad.speaking)
        self.assertTrue(vad._pending_start)
        self.assertIsNone(vad.apply_event("inference_done", now=clock + 0.2))
        self.assertEqual(vad.apply_event("inference_done", now=clock + 0.51), "start")
        self.assertTrue(vad.speaking)

    def test_echo_blip_cancelled_by_end(self) -> None:
        vad = LiveKitSileroVad()
        vad.bot_playing = True
        self.assertIsNone(vad.apply_event("start_of_speech", now=1.0))
        self.assertIsNone(vad.apply_event("end_of_speech", now=1.1))
        self.assertFalse(vad.speaking)
        self.assertFalse(vad._pending_start)

    def test_energy_assist_commits_when_silero_is_silent(self) -> None:
        import barge_in as mod

        vad = LiveKitSileroVad(start_secs=0.15, stop_secs=0.20)
        loud = array.array("h", [8000] * 320).tobytes()
        quiet = array.array("h", [20] * 320).tobytes()
        clock = [0.0]
        orig = mod.time.monotonic
        mod.time.monotonic = lambda: clock[0]
        try:
            edge = None
            for i in range(20):
                clock[0] = i * 0.02
                got = vad.feed(loud)
                if got:
                    edge = got
                    break
            self.assertEqual(edge, "start")
            self.assertTrue(vad._assist_active)
            stop = None
            for i in range(20, 50):
                clock[0] = i * 0.02
                got = vad.feed(quiet)
                if got:
                    stop = got
                    break
            self.assertEqual(stop, "stop")
            self.assertFalse(vad.speaking)
        finally:
            mod.time.monotonic = orig


class _FakeWs:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))


def _pcm_chunk(sample: int) -> bytes:
    n = 16_000  # 1 s at MODEL_RATE
    return array.array("h", [int(sample)] * n).tobytes()


class NativeEngineTests(unittest.IsolatedAsyncioTestCase):
    def test_pcm_float_roundtrip(self) -> None:
        import array

        pcm = array.array("h", [0, 16384, -16384, 32767, -32768]).tobytes()
        raw = _pcm16_to_f32le(pcm)
        back = _f32le_to_pcm16(raw)
        src = array.array("h")
        src.frombytes(pcm)
        dst = array.array("h")
        dst.frombytes(back)
        self.assertEqual(len(src), len(dst))
        for a, b in zip(src, dst):
            self.assertLessEqual(abs(a - b), 1)

    def test_worker_url_remaps_vllm_port(self) -> None:
        old_base = os.environ.get("MODEL_BASE_URL")
        old_native = os.environ.get("NATIVE_WORKER_URL")
        os.environ["MODEL_BASE_URL"] = "http://127.0.0.1:8199"
        os.environ.pop("NATIVE_WORKER_URL", None)
        try:
            self.assertEqual(_worker_ws_url(), "ws://127.0.0.1:22400/v1/worker/duplex")
            os.environ["MODEL_BASE_URL"] = "http://127.0.0.1:22400"
            self.assertEqual(_worker_ws_url(), "ws://127.0.0.1:22400/v1/worker/duplex")
            os.environ["NATIVE_WORKER_URL"] = "http://10.0.0.9:22400"
            self.assertEqual(_worker_ws_url(), "ws://10.0.0.9:22400/v1/worker/duplex")
        finally:
            if old_base is None:
                os.environ.pop("MODEL_BASE_URL", None)
            else:
                os.environ["MODEL_BASE_URL"] = old_base
            if old_native is None:
                os.environ.pop("NATIVE_WORKER_URL", None)
            else:
                os.environ["NATIVE_WORKER_URL"] = old_native

    async def test_listen_keeps_queued_audio(self) -> None:
        eng = NativeMiniCPMEngine("hello")
        eng._speaking = True
        await eng._out.put((b"\x00\x00", 24000))
        await eng._on_event({"type": "response.output.delta", "kind": "listen"})
        self.assertFalse(eng.generating)
        self.assertEqual(eng._out.qsize(), 1)

    async def test_listen_lock_reconnects_while_user_talks(self) -> None:
        eng = NativeMiniCPMEngine("hello")
        eng._last_append_rms = 900
        for _ in range(LISTEN_LOCK_N):
            await eng._on_event({"type": "response.output.delta", "kind": "listen"})
        self.assertEqual(eng._reconnect_reason, "listen_lock")
        self.assertEqual(eng._listen_speech_streak, LISTEN_LOCK_N)

    async def test_interrupt_sets_force_listen_and_drops_queue(self) -> None:
        eng = NativeMiniCPMEngine("hello")
        eng._speaking = True
        await eng._out.put((b"\x00\x00", 24000))
        await eng.interrupt()
        self.assertTrue(eng._force_listen)
        self.assertFalse(eng.generating)
        self.assertEqual(eng._out.qsize(), 0)
        self.assertFalse(hasattr(eng, "end_of_speech"))

    async def test_speak_yield_force_listens_after_ramble(self) -> None:
        eng = NativeMiniCPMEngine("hello")
        eng._ws = _FakeWs()
        eng._speaking = True
        eng._speak_streak = SPEAK_YIELD_N
        await eng._out.put((b"\x00\x00", 24000))
        await eng._append(_pcm_chunk(0))
        self.assertFalse(eng.generating)
        self.assertEqual(eng._out.qsize(), 0)
        payload = eng._ws.sent[-1]
        self.assertTrue(payload["input"]["force_listen"])

    async def test_speak_barge_force_listens_while_user_talks(self) -> None:
        eng = NativeMiniCPMEngine("hello")
        eng._ws = _FakeWs()
        eng._speaking = True
        loud = _pcm_chunk(900)
        await eng._append(loud)
        self.assertFalse(eng._ws.sent[-1]["input"]["force_listen"])
        self.assertEqual(eng._speak_user_streak, 1)
        eng._speaking = True
        await eng._append(loud)
        self.assertGreaterEqual(SPEAK_BARGE_N, 2)
        self.assertTrue(eng._ws.sent[-1]["input"]["force_listen"])
        self.assertFalse(eng.generating)

    def test_qwen_native_refused(self) -> None:
        from agent import SpeechSession

        old_backend = os.environ.get("MODEL_BACKEND")
        old_model = os.environ.get("MODEL_NAME")
        os.environ["MODEL_BACKEND"] = "native"
        os.environ["MODEL_NAME"] = "Qwen/Qwen2.5-Omni-3B"
        try:
            with self.assertRaises(RuntimeError) as ctx:
                SpeechSession("default")
            self.assertIn("minicpm", str(ctx.exception).lower())
        finally:
            if old_backend is None:
                os.environ.pop("MODEL_BACKEND", None)
            else:
                os.environ["MODEL_BACKEND"] = old_backend
            if old_model is None:
                os.environ.pop("MODEL_NAME", None)
            else:
                os.environ["MODEL_NAME"] = old_model


class PlayoutGapTests(unittest.TestCase):
    def test_generating_gap_does_not_rebuffer(self) -> None:
        import array

        from audio import WEBRTC_RATE, ClockedPlayout

        p = ClockedPlayout()
        p.source_active = True
        p._buffering = False
        pcm = array.array("h", [8000] * (WEBRTC_RATE // 5)).tobytes()  # 200 ms
        t = 1000.0
        p.push(pcm, WEBRTC_RATE)
        p._last_push = t
        for i in range(12):
            frame = p.take_frame(t + i * 0.02)
            self.assertIsNotNone(frame)
        gap = p.take_frame(t + 0.50)
        self.assertIsNotNone(gap)
        self.assertFalse(p._buffering)
        self.assertEqual(p.underruns_total, 0)
        p.push(pcm, WEBRTC_RATE)
        nxt = p.take_frame(t + 0.52)
        self.assertIsNotNone(nxt)
        self.assertFalse(p._buffering)
        self.assertEqual(p.underruns_total, 0)


if __name__ == "__main__":
    os.chdir(ROOT)
    unittest.main()
