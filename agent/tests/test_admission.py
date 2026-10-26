"""Greenroom admission: hold Connect until the live speech socket accepts."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _token_request(voice: str = "original") -> object:
    from fastapi import Request

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/livekit/token",
        "raw_path": b"/api/livekit/token",
        "query_string": f"role=patient&room=default&voice={voice}".encode(),
        "headers": [],
        "client": ("127.0.0.1", 9),
        "server": ("test", 80),
    }
    return Request(scope)


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        import token_server as ts

        self.ts = ts
        self._bots = dict(ts._bots)
        self._humans = {k: set(v) for k, v in ts._humans.items()}
        self._ready_cache = ts._ready_cache
        ts._bots.clear()
        ts._humans.clear()
        ts._ready_cache = None
        self._old_model = os.environ.get("MODEL_NAME")
        self._old_backend = os.environ.get("MODEL_BACKEND")
        os.environ["MODEL_NAME"] = "cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit"
        os.environ["MODEL_BACKEND"] = "vllm"

    async def asyncTearDown(self) -> None:
        self.ts._bots.clear()
        self.ts._humans.clear()
        self.ts._bots.update(self._bots)
        self.ts._humans.update(self._humans)
        self.ts._ready_cache = self._ready_cache
        if self._old_model is None:
            os.environ.pop("MODEL_NAME", None)
        else:
            os.environ["MODEL_NAME"] = self._old_model
        if self._old_backend is None:
            os.environ.pop("MODEL_BACKEND", None)
        else:
            os.environ["MODEL_BACKEND"] = self._old_backend

    async def test_holds_while_stream_closed(self) -> None:
        probe = {
            "tcp_ok": True,
            "http_ok": True,
            "models_ok": True,
            "detail": "models=ok",
        }
        with patch.object(self.ts, "livekit_tcp_ok", new=AsyncMock(return_value=(True, "ok"))):
            with patch.object(self.ts, "probe_model_server", new=AsyncMock(return_value=probe)):
                with patch.object(
                    self.ts,
                    "probe_video_stream",
                    new=AsyncMock(return_value={"stream_ok": False, "stream_detail": "refused"}),
                ):
                    snap = await self.ts.admission_status("default")
        self.assertFalse(snap["admit"])
        self.assertEqual(snap["phase"], "waiting_engine")
        self.assertIn("not accepting", snap["message"])

    async def test_admits_when_stream_accepts(self) -> None:
        probe = {
            "tcp_ok": True,
            "http_ok": True,
            "models_ok": True,
            "detail": "models=ok",
        }
        with patch.object(self.ts, "livekit_tcp_ok", new=AsyncMock(return_value=(True, "ok"))):
            with patch.object(self.ts, "probe_model_server", new=AsyncMock(return_value=probe)):
                with patch.object(
                    self.ts,
                    "probe_video_stream",
                    new=AsyncMock(return_value={"stream_ok": True, "stream_detail": "accepted"}),
                ):
                    snap = await self.ts.admission_status("default")
        self.assertTrue(snap["admit"])
        self.assertEqual(snap["phase"], "ready")

    async def test_admits_when_engine_already_ready(self) -> None:
        probe = {
            "tcp_ok": True,
            "http_ok": True,
            "models_ok": True,
            "detail": "models=ok",
        }
        self.ts._bots["default"] = SimpleNamespace(
            _running=True,
            engine=SimpleNamespace(ready=True),
        )
        with patch.object(self.ts, "livekit_tcp_ok", new=AsyncMock(return_value=(True, "ok"))):
            with patch.object(self.ts, "probe_model_server", new=AsyncMock(return_value=probe)):
                with patch.object(
                    self.ts,
                    "probe_video_stream",
                    new=AsyncMock(return_value={"stream_ok": False, "stream_detail": "busy"}),
                ):
                    snap = await self.ts.admission_status("default")
        self.assertTrue(snap["admit"])
        self.assertEqual(snap["phase"], "ready")
        self.assertTrue(snap["engine_ready"])

    async def test_token_503_while_greenroom_holds(self) -> None:
        os.environ["MODEL_NAME"] = "openbmb/MiniCPM-o-4_5"
        hold = {
            "admit": False,
            "phase": "waiting_engine",
            "message": "Live speech socket is not accepting yet. Almost there.",
        }
        request = _token_request()
        with patch.object(self.ts, "livekit_tcp_ok", new=AsyncMock(return_value=(True, "ok"))):
            with patch.object(self.ts, "admission_status", new=AsyncMock(return_value=hold)):
                resp = await self.ts.livekit_token(request)
        self.assertEqual(resp.status_code, 503)
        self.assertFalse(self.ts._humans.get("default"))
        self.assertFalse(self.ts._bots)
        self.assertIn("greenroom", resp.body.decode())

    async def test_health_stays_ok_during_load(self) -> None:
        """Docker healthcheck must not restart the agent while weights load."""
        with patch.object(self.ts, "livekit_tcp_ok", new=AsyncMock(return_value=(True, "ok"))):
            with patch.object(self.ts, "model_server_tcp_ok", new=AsyncMock(return_value=(True, "8199"))):
                body = await self.ts.health()
        self.assertTrue(body["ok"])
        self.assertFalse(body["admit"])
        self.assertIn(body["phase"], ("up", "waiting_sfu", "waiting_model", "waiting_engine", "ready"))

    async def test_native_needs_model_loaded(self) -> None:
        os.environ["MODEL_BACKEND"] = "native"
        os.environ["MODEL_NAME"] = "openbmb/MiniCPM-o-4_5"
        worker = {
            "tcp_ok": True,
            "http_ok": True,
            "model_loaded": False,
            "detail": "loading",
        }
        with patch.object(self.ts, "livekit_tcp_ok", new=AsyncMock(return_value=(True, "ok"))):
            with patch.object(self.ts, "probe_native_worker", new=AsyncMock(return_value=worker)):
                snap = await self.ts.admission_status("default")
        self.assertFalse(snap["admit"])
        self.assertEqual(snap["phase"], "waiting_model")


class ProbeVideoStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_closes_after_accept(self) -> None:
        import qwen_omni as qwen_omni_mod

        closed = {"n": 0}

        class FakeWs:
            async def close(self) -> None:
                closed["n"] += 1

        async def fake_connect(*_a, **_k):
            return FakeWs()

        with patch.object(qwen_omni_mod.websockets, "connect", new=fake_connect):
            out = await qwen_omni_mod.probe_video_stream(timeout_s=0.2)
        self.assertTrue(out["stream_ok"])
        self.assertEqual(closed["n"], 1)

    async def test_reports_refusal(self) -> None:
        import qwen_omni as qwen_omni_mod

        async def boom(*_a, **_k):
            raise OSError("connection refused")

        with patch.object(qwen_omni_mod.websockets, "connect", new=boom):
            out = await qwen_omni_mod.probe_video_stream(timeout_s=0.2)
        self.assertFalse(out["stream_ok"])
        self.assertIn("refused", out["stream_detail"])


if __name__ == "__main__":
    unittest.main()
