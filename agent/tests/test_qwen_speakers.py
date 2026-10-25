"""Qwen3-Omni built-in speakers: allowlist only, no WAV validation."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen_speakers import (  # noqa: E402
    DEFAULT_QWEN_SPEAKER,
    QWEN_SPEAKER_ENV,
    QwenSpeakerNotFound,
    default_qwen_speaker_id,
    list_qwen_speakers,
    qwen_speakers_public,
    resolve_qwen_speaker,
)


class QwenSpeakerCatalogTests(unittest.TestCase):
    def test_list_has_three_speakers(self) -> None:
        speakers = list_qwen_speakers()
        self.assertEqual(len(speakers), 3)
        ids = {s.id for s in speakers}
        self.assertEqual(ids, {"chelsie", "ethan", "aiden"})

    def test_resolve_valid_speakers(self) -> None:
        for name in ("chelsie", "Chelsie", "CHELSIE", "ethan", "Ethan", "aiden", "Aiden"):
            speaker = resolve_qwen_speaker(name)
            self.assertEqual(speaker.id, name.lower())

    def test_resolve_empty_uses_default(self) -> None:
        os.environ.pop(QWEN_SPEAKER_ENV, None)
        speaker = resolve_qwen_speaker("")
        self.assertEqual(speaker.id, DEFAULT_QWEN_SPEAKER)

    def test_resolve_env_override(self) -> None:
        os.environ[QWEN_SPEAKER_ENV] = "aiden"
        try:
            speaker = resolve_qwen_speaker("")
            self.assertEqual(speaker.id, "aiden")
        finally:
            os.environ.pop(QWEN_SPEAKER_ENV, None)

    def test_resolve_explicit_beats_env(self) -> None:
        os.environ[QWEN_SPEAKER_ENV] = "aiden"
        try:
            speaker = resolve_qwen_speaker("chelsie")
            self.assertEqual(speaker.id, "chelsie")
        finally:
            os.environ.pop(QWEN_SPEAKER_ENV, None)

    def test_reject_unknown_speaker(self) -> None:
        for bad in ("original", "heart", "unknown", "minicpm", "ryan"):
            with self.assertRaises(QwenSpeakerNotFound):
                resolve_qwen_speaker(bad)

    def test_public_catalog_format(self) -> None:
        pub = qwen_speakers_public()
        self.assertEqual(len(pub), 3)
        for item in pub:
            self.assertIn("id", item)
            self.assertIn("name", item)
            self.assertIn("gender", item)
            self.assertIn("blurb", item)
            self.assertIn("featured", item)
            # No WAV fields
            self.assertNotIn("source", item)
            self.assertNotIn("license", item)
            self.assertNotIn("filename", item)

    def test_default_speaker_id_env(self) -> None:
        os.environ[QWEN_SPEAKER_ENV] = "ethan"
        try:
            self.assertEqual(default_qwen_speaker_id(), "ethan")
        finally:
            os.environ.pop(QWEN_SPEAKER_ENV, None)

    def test_default_speaker_id_fallback(self) -> None:
        os.environ.pop(QWEN_SPEAKER_ENV, None)
        self.assertEqual(default_qwen_speaker_id(), DEFAULT_QWEN_SPEAKER)


if __name__ == "__main__":
    unittest.main()
