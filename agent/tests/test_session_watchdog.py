"""Client-only turn watchdog: recover dead duplex sessions without protocol changes."""

from __future__ import annotations

import asyncio
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
from barge_in import EnergyVad, VAD_START_SECS, VAD_STOP_SECS  # noqa: E402


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


if __name__ == "__main__":
    os.chdir(ROOT)
    unittest.main()
