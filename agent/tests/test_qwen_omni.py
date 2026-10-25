"""Qwen client helpers: MiniCPM prompt must not go on the video-stream socket."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen_omni import QwenOmniEngine, _stall_s, _system_prompt  # noqa: E402
from qwen_speakers import DEFAULT_QWEN_SPEAKER  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
