"""Qwen client helpers: MiniCPM prompt must not go on the video-stream socket."""

from __future__ import annotations

import array
import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import qwen_omni as qwen_omni_mod  # noqa: E402
from qwen_omni import QwenOmniEngine, _stall_s, _system_prompt  # noqa: E402
from qwen_speakers import DEFAULT_QWEN_SPEAKER  # noqa: E402


class _FakeWs:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))


def _speech_pcm(seconds: float = 1.0, sample: int = 2000) -> bytes:
    n = int(16_000 * seconds)
    return array.array("h", [int(sample)] * n).tobytes()


async def _quiet_probe() -> dict:
    return {
        "model_base_url": "http://127.0.0.1:8199",
        "host": "127.0.0.1",
        "port": 8199,
        "tcp_ok": False,
        "http_ok": False,
        "models_ok": False,
        "detail": "test",
    }


class QwenPromptTests(unittest.TestCase):
    def test_replaces_minicpm_clone_prompt(self) -> None:
        raw = (
            "Clone the voice in the provided audio prompt. Please assist users "
            "while maintaining this voice style. You are a helpful assistant "
            "developed by ModelBest: MiniCPM-Omni."
        )
        out = _system_prompt(raw)
        self.assertNotIn("MiniCPM", out)
        self.assertNotIn("Clone the voice", out)
        self.assertIn("spoken English", out)

    def test_keeps_custom_prompt(self) -> None:
        self.assertEqual(_system_prompt("Be brief."), "Be brief.")

    def test_empty_uses_default(self) -> None:
        self.assertIn("camera", _system_prompt(""))

    def test_engine_applies_prompt(self) -> None:
        eng = QwenOmniEngine("Clone the voice in the provided audio prompt. MiniCPM.")
        self.assertNotIn("MiniCPM", eng.system_prompt)


class QwenStallTests(unittest.TestCase):
    def test_default_stall(self) -> None:
        os.environ.pop("QWEN_STALL_S", None)
        self.assertEqual(_stall_s(), 12.0)

    def test_floor(self) -> None:
        os.environ["QWEN_STALL_S"] = "1"
        try:
            self.assertEqual(_stall_s(), 4.0)
        finally:
            os.environ.pop("QWEN_STALL_S", None)

    def test_finish_turn_clears_busy(self) -> None:
        eng = QwenOmniEngine()
        eng._busy = True
        eng._t_query = 1.0
        eng._finish_turn("stall_no_audio")
        self.assertFalse(eng.generating)
        self.assertEqual(eng._t_query, 0.0)


class QwenSpeakerTests(unittest.TestCase):
    def test_speaker_explicit(self) -> None:
        eng = QwenOmniEngine("test", speaker="aiden")
        self.assertEqual(eng.speaker, "aiden")

    def test_speaker_from_env(self) -> None:
        os.environ["QWEN_SPEAKER"] = "Ethan"
        try:
            eng = QwenOmniEngine("test")
            self.assertEqual(eng.speaker, "ethan")
        finally:
            os.environ.pop("QWEN_SPEAKER", None)

    def test_speaker_explicit_beats_env(self) -> None:
        os.environ["QWEN_SPEAKER"] = "Ethan"
        try:
            eng = QwenOmniEngine("test", speaker="Chelsie")
            self.assertEqual(eng.speaker, "chelsie")
        finally:
            os.environ.pop("QWEN_SPEAKER", None)

    def test_speaker_default(self) -> None:
        os.environ.pop("QWEN_SPEAKER", None)
        eng = QwenOmniEngine("test")
        self.assertEqual(eng.speaker, DEFAULT_QWEN_SPEAKER)

    def test_speaker_in_snapshot(self) -> None:
        eng = QwenOmniEngine("test", speaker="aiden")
        snap = eng.stream_snapshot()
        self.assertEqual(snap["speaker"], "aiden")

    def test_session_config_includes_speaker(self) -> None:
        eng = QwenOmniEngine("test", speaker="Ethan")
        cfg = eng._session_config()
        self.assertEqual(cfg["type"], "session.config")
        self.assertEqual(cfg["speaker"], "ethan")


class QwenPendingTurnTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._orig_probe = qwen_omni_mod.probe_model_server
        qwen_omni_mod.probe_model_server = _quiet_probe

    async def asyncTearDown(self) -> None:
        qwen_omni_mod.probe_model_server = self._orig_probe

    async def test_generate_while_not_ready_is_queued_then_flushed(self) -> None:
        eng = QwenOmniEngine("test", speaker="chelsie")
        await eng.push_audio(_speech_pcm(1.0), 16_000, is_speech=True)
        await eng.push_image(b"jpeg-bytes")
        await eng.generate()
        self.assertTrue(eng._pending_generate)
        self.assertGreater(len(eng._pending_pcm), 0)
        self.assertEqual(eng._pending_jpeg, b"jpeg-bytes")
        self.assertFalse(eng.ready)

        ws = _FakeWs()
        eng._ws = ws
        eng._ready.set()
        await eng._flush_pending_after_ready()
        types = [row.get("type") for row in ws.sent]
        self.assertIn("audio.chunk", types)
        self.assertIn("video.frame", types)
        self.assertIn("video.query", types)
        self.assertFalse(eng._pending_generate)
        self.assertEqual(len(eng._pending_pcm), 0)
        self.assertTrue(eng.generating)

    async def test_flush_waits_if_user_is_still_speaking(self) -> None:
        eng = QwenOmniEngine("test")
        await eng.push_audio(_speech_pcm(1.0), 16_000, is_speech=True)
        await eng.push_image(b"jpeg-bytes")
        ws = _FakeWs()
        eng._ws = ws
        eng._ready.set()
        self.assertTrue(eng._speech_open)
        await eng._flush_pending_after_ready()
        self.assertNotIn("video.query", [row.get("type") for row in ws.sent])
        self.assertFalse(eng._pending_generate)

        await eng.generate()
        self.assertIn("video.query", [row.get("type") for row in ws.sent])

    async def test_generate_promotes_stashed_media_when_already_ready(self) -> None:
        eng = QwenOmniEngine("test")
        await eng.push_audio(_speech_pcm(1.0), 16_000, is_speech=True)
        await eng.push_image(b"jpeg-bytes")
        ws = _FakeWs()
        eng._ws = ws
        eng._ready.set()
        await eng.generate()
        types = [row.get("type") for row in ws.sent]
        self.assertIn("audio.chunk", types)
        self.assertIn("video.frame", types)
        self.assertIn("video.query", types)

    async def test_snapshot_exposes_pending_audio(self) -> None:
        eng = QwenOmniEngine("test")
        await eng.push_audio(_speech_pcm(1.0), 16_000, is_speech=True)
        await eng.generate()
        snap = eng.stream_snapshot()
        self.assertTrue(snap["pending_generate"])
        self.assertGreaterEqual(snap["pending_audio_ms"], 900)
        self.assertGreaterEqual(snap["utt_ms"], 900)

    async def test_interrupt_clears_pending_hello(self) -> None:
        eng = QwenOmniEngine("test")
        await eng.push_audio(_speech_pcm(1.0), 16_000, is_speech=True)
        await eng.generate()
        await eng.interrupt()
        self.assertFalse(eng._pending_generate)
        self.assertEqual(len(eng._pending_pcm), 0)


if __name__ == "__main__":
    unittest.main()
