"""Allowlisted MiniCPM voices: catalog, WAV checks, and room conflict."""

from __future__ import annotations

import array
import hashlib
import io
import math
import os
import sys
import unittest
import wave
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voices import (  # noqa: E402
    DEFAULT_VOICE_ID,
    VoiceNotFound,
    asset_path,
    catalog_public,
    resolve_voice,
    validate_catalog_voice,
    validate_wav,
    voice_conflict,
)
from duplex_model import MiniCPMDuplexEngine, _ref_audio_data_url  # noqa: E402


def _pcm_wav(seconds: float = 3.2, rate: int = 16000, channels: int = 1, sampwidth: int = 2) -> bytes:
    frames = int(rate * seconds)
    payload = b"\x00" * (frames * channels * sampwidth)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sampwidth)
        handle.setframerate(rate)
        handle.writeframes(payload)
    return buf.getvalue()


class VoiceCatalogTests(unittest.TestCase):
    def test_default_is_original_without_asset(self) -> None:
        voice = resolve_voice("")
        self.assertEqual(voice.id, DEFAULT_VOICE_ID)
        self.assertIsNone(voice.filename)
        self.assertIsNone(asset_path(voice))
        info = validate_catalog_voice(voice)
        self.assertTrue(info["bundled"])

    def test_unknown_and_path_escape_rejected(self) -> None:
        with self.assertRaises(VoiceNotFound):
            resolve_voice("not-a-voice")
        with self.assertRaises(VoiceNotFound):
            resolve_voice("../heart")
        with self.assertRaises(VoiceNotFound):
            resolve_voice("/etc/passwd")

    def test_public_catalog_has_no_paths(self) -> None:
        blob = str(catalog_public())
        self.assertNotIn("/home", blob)
        self.assertNotIn("assets/voices", blob)
        ids = {row["id"] for row in catalog_public()}
        self.assertIn("original", ids)
        self.assertIn("heart", ids)
        self.assertIn("ryan", ids)

    def test_catalog_wavs_validate(self) -> None:
        for vid in ("heart", "bella", "nicole", "michael", "ryan", "joe", "lessac", "amy"):
            voice = resolve_voice(vid)
            info = validate_catalog_voice(voice)
            path = asset_path(voice)
            self.assertIsNotNone(path)
            assert path is not None
            self.assertEqual(info["rate"], 16000)
            self.assertEqual(info["channels"], 1)
            self.assertGreaterEqual(info["duration_s"], 3.0)
            self.assertLessEqual(info["duration_s"], 6.5)
            with wave.open(str(path), "rb") as handle:
                samples = array.array("h")
                samples.frombytes(handle.readframes(handle.getnframes()))
            peak = max(abs(x) for x in samples)
            peak_db = 20 * math.log10(peak / 32768)
            self.assertLessEqual(peak_db, -5.0)
            self.assertGreaterEqual(peak_db, -7.0)
            self.assertEqual(info["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_validate_wav_rejects_bad_format(self) -> None:
        tmp = Path(self._tmp())
        tmp.write_bytes(_pcm_wav(seconds=1.0))
        with self.assertRaises(ValueError):
            validate_wav(tmp)
        tmp.write_bytes(_pcm_wav(seconds=3.2, channels=2))
        with self.assertRaises(ValueError):
            validate_wav(tmp)
        tmp.write_bytes(b"not-a-wav")
        with self.assertRaises(ValueError):
            validate_wav(tmp)

    def test_engine_keeps_selected_ref_on_instance(self) -> None:
        voice = resolve_voice("heart")
        path = asset_path(voice)
        eng = MiniCPMDuplexEngine("hello", voice_id=voice.id, ref_path=path)
        self.assertEqual(eng.voice_id, "heart")
        self.assertEqual(eng._ref_path, path)
        snap = eng.stream_snapshot()
        self.assertEqual(snap["voice_id"], "heart")
        eng._reset_stream_state()
        self.assertEqual(eng.voice_id, "heart")
        self.assertEqual(eng._ref_path, path)

    def test_original_engine_uses_env_or_bundled_lookup(self) -> None:
        eng = MiniCPMDuplexEngine()
        self.assertEqual(eng.voice_id, "original")
        self.assertIsNone(eng._ref_path)

    def test_original_handshake_ignores_clone_ref_path(self) -> None:
        voice = resolve_voice("heart")
        path = asset_path(voice)
        original = MiniCPMDuplexEngine("hello", voice_id="original", ref_path=path)
        self.assertIsNone(original._handshake_ref_path())
        clone = MiniCPMDuplexEngine("hello", voice_id="heart", ref_path=path)
        self.assertEqual(clone._handshake_ref_path(), path)

    def test_original_duplex_handshake_still_locked(self) -> None:
        src = (Path(__file__).resolve().parents[1] / "src" / "duplex_model.py").read_text(encoding="utf-8")
        for token in (
            '"overlap_policy": "listen_only"',
            '"playback_commit_policy": "ack_only"',
            '"auto_response": True',
            '"force_listen_count": 0',
            '"sliding_window_mode": "basic"',
            '"basic_window_high_tokens": 3000',
            '"basic_window_low_tokens": 2000',
        ):
            self.assertIn(token, src)
        yaml = Path(__file__).resolve().parents[2] / "model-server" / "deploy" / "minicpmo_4_5.yaml"
        text = yaml.read_text(encoding="utf-8")
        self.assertIn("codec_chunk_frames: 24", text)
        self.assertIn("enable_cfm_graph: false", text)

    def test_ref_loader_reads_explicit_path(self) -> None:
        tmp = Path(self._tmp())
        tmp.write_bytes(_pcm_wav(seconds=3.2))
        url, source, meta = _ref_audio_data_url(tmp)
        self.assertTrue(url.startswith("data:audio/wav;base64,"))
        self.assertIn(tmp.name, source)
        self.assertGreater(meta["bytes"], 64)
        self.assertEqual(len(meta["sha256"]), 12)

    def test_conflict_helper(self) -> None:
        class Bot:
            def __init__(self, voice_id: str, running: bool = True) -> None:
                self.voice_id = voice_id
                self._running = running

        self.assertIsNone(voice_conflict(None, "heart"))
        self.assertIsNone(voice_conflict(Bot("heart", running=False), "original"))
        self.assertIsNone(voice_conflict(Bot("heart"), "heart"))
        self.assertEqual(voice_conflict(Bot("original"), "heart"), "original")

    def _tmp(self) -> str:
        import tempfile

        handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        handle.close()
        self.addCleanup(lambda: os.path.exists(handle.name) and os.unlink(handle.name))
        return handle.name


class TokenVoiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        import token_server as ts

        self.ts = ts
        self._bots = dict(ts._bots)
        self._humans = {k: set(v) for k, v in ts._humans.items()}
        ts._bots.clear()
        ts._humans.clear()
        # Force MiniCPM mode for these tests (voice catalog tests)
        self._old_model = os.environ.get("MODEL_NAME")
        os.environ["MODEL_NAME"] = "openbmb/MiniCPM-o-4_5"

    async def asyncTearDown(self) -> None:
        self.ts._bots.clear()
        self.ts._humans.clear()
        self.ts._bots.update(self._bots)
        self.ts._humans.update(self._humans)
        # Restore MODEL_NAME
        if self._old_model is None:
            os.environ.pop("MODEL_NAME", None)
        else:
            os.environ["MODEL_NAME"] = self._old_model

    async def test_unknown_voice_does_not_register_human(self) -> None:
        from fastapi import Request

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/livekit/token",
            "raw_path": b"/api/livekit/token",
            "query_string": b"role=patient&room=default&voice=nope",
            "headers": [],
            "client": ("127.0.0.1", 9),
            "server": ("test", 80),
        }
        request = Request(scope)
        resp = await self.ts.livekit_token(request)
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(self.ts._humans.get("default"))
        self.assertFalse(self.ts._bots)

    async def test_conflict_does_not_register_human(self) -> None:
        class Bot:
            voice_id = "original"
            _running = True

        self.ts._bots["default"] = Bot()
        from fastapi import Request

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/livekit/token",
            "raw_path": b"/api/livekit/token",
            "query_string": b"role=patient&room=default&voice=heart",
            "headers": [],
            "client": ("127.0.0.1", 9),
            "server": ("test", 80),
        }
        request = Request(scope)
        with patch.object(self.ts, "livekit_tcp_ok", new=AsyncMock(return_value=(True, "ok"))):
            with patch.object(self.ts, "token_for_user", return_value="tok"):
                resp = await self.ts.livekit_token(request)
        self.assertEqual(resp.status_code, 409)
        self.assertFalse(self.ts._humans.get("default"))
        self.assertIs(self.ts._bots["default"], self.ts._bots.get("default"))
        self.assertEqual(self.ts._bots["default"].voice_id, "original")


if __name__ == "__main__":
    unittest.main()
