"""Session lab: snapshot contracts, scoreboard, and /api/live shape."""

from __future__ import annotations

import array
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio import PLAYOUT_START_MS, WEBRTC_RATE, ClockedPlayout  # noqa: E402
from duplex_model import MiniCPMDuplexEngine  # noqa: E402
from native_minicpm import NativeMiniCPMEngine  # noqa: E402
from qwen_omni import QwenOmniEngine  # noqa: E402
from session_score import SessionScore, TurnSample  # noqa: E402


class SessionScoreTests(unittest.TestCase):
    def test_percentiles_and_snapshot(self) -> None:
        score = SessionScore()
        score.add_turn(TurnSample(ttfa_ms=100, rtf=1.0, prebuffer_ms=150, e2e_ms=250))
        score.add_turn(TurnSample(ttfa_ms=200, rtf=1.2, prebuffer_ms=140, e2e_ms=340))
        score.add_turn(TurnSample(ttfa_ms=400, rtf=0.8, prebuffer_ms=160, e2e_ms=560))
        snap = score.snapshot()
        self.assertEqual(snap["n"], 3)
        self.assertEqual(snap["ttfa"]["p50"], 200.0)
        self.assertGreaterEqual(snap["ttfa"]["p95"], snap["ttfa"]["p50"])
        self.assertEqual(snap["last_turn"]["ttfa_ms"], 400)

    def test_ignores_unset_values(self) -> None:
        score = SessionScore()
        score.add_turn(TurnSample(ttfa_ms=-1, rtf=-1))
        snap = score.snapshot()
        self.assertEqual(snap["ttfa"]["p50"], -1.0)
        self.assertEqual(snap["rtf"]["mean"], -1.0)


class SnapshotContractTests(unittest.TestCase):
    """Existing floor strip and /api/live still read these keys."""

    def test_qwen_keeps_floor_keys(self) -> None:
        snap = QwenOmniEngine("test", speaker="ethan").stream_snapshot()
        for key in ("ready", "speaking", "out_q", "frame_n", "utt_ms", "delta_age_ms", "speaker", "rtf", "pending_generate"):
            self.assertIn(key, snap, key)
        self.assertEqual(snap["ttfa_anchor"], "query")

    def test_duplex_keeps_floor_keys(self) -> None:
        snap = MiniCPMDuplexEngine().stream_snapshot()
        for key in ("ready", "speaking", "out_q", "turn_waiting", "rtf", "buf_ms", "voice_id"):
            self.assertIn(key, snap, key)
        self.assertEqual(snap["ttfa_anchor"], "speak")

    def test_native_keeps_floor_keys(self) -> None:
        snap = NativeMiniCPMEngine().stream_snapshot()
        for key in ("ready", "speaking", "out_q", "kv", "cost_all_ms_p50", "voice_id"):
            self.assertIn(key, snap, key)
        self.assertEqual(snap["ttfa_anchor"], "speak-start")


class PlayoutPrebufferTests(unittest.TestCase):
    def test_prebuffer_survives_first_frame(self) -> None:
        p = ClockedPlayout()
        p.source_active = True
        need = int(WEBRTC_RATE * PLAYOUT_START_MS / 1000) + 480
        pcm = array.array("h", [8000] * need).tobytes()
        t = 10.0
        p.push(pcm, WEBRTC_RATE)
        p.push(pcm, WEBRTC_RATE)
        frame = p.take_frame(t + PLAYOUT_START_MS / 1000.0)
        self.assertIsNotNone(frame)
        stats = p.turn_stats()
        self.assertGreaterEqual(stats["prebuffer_ms"], 0.0)
        self.assertFalse(p._buffering)

    def test_close_turn_does_not_raise(self) -> None:
        """Metrics refactor deleted _emit_turn_summary; idle/clear must still work."""
        p = ClockedPlayout()
        p.source_active = False
        need = int(WEBRTC_RATE * PLAYOUT_START_MS / 1000) + 480
        pcm = array.array("h", [8000] * need).tobytes()
        p.push(pcm, WEBRTC_RATE)
        self.assertIsNotNone(p.take_frame())
        p._close_turn("idle")
        p.clear()


class ApiLiveContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_top_level_rtf_still_present(self) -> None:
        from token_server import _bots, api_live

        class Engine:
            generating = False
            ready = True

            def stream_snapshot(self) -> dict:
                return {
                    "ready": True,
                    "speaking": False,
                    "out_q": 0,
                    "rtf": 1.25,
                    "turn_waiting": False,
                    "buf_ms": 40,
                    "speaker": "ethan",
                    "ttfa_ms": 180,
                    "ttfa_anchor": "speak",
                    "ttft_ms": 90,
                    "gap_ms": 40,
                    "turn_audio_ms": 800,
                }

        bot = SimpleNamespace(
            _running=True,
            voice_id="ethan",
            engine=Engine(),
            vad=SimpleNamespace(speaking=False),
            playout=SimpleNamespace(audible=False, ring_ms=12.5),
            session_score_snapshot=lambda: {"n": 0},
        )
        room = "lab-contract"
        prev = _bots.get(room)
        _bots[room] = bot  # type: ignore[assignment]
        try:
            request = SimpleNamespace(query_params={"room": room})
            body = await api_live(request)  # type: ignore[arg-type]
        finally:
            if prev is None:
                _bots.pop(room, None)
            else:
                _bots[room] = prev

        self.assertTrue(body["ok"])
        self.assertEqual(body["rtf"], 1.25)
        self.assertEqual(body["turn_waiting"], False)
        self.assertEqual(body["buf_ms"], 40)
        self.assertEqual(body["speaker"], "ethan")
        self.assertEqual(body["ring_ms"], 12.5)
        self.assertEqual(body["turn"]["ttfa_ms"], 180)
        self.assertEqual(body["session"]["n"], 0)


if __name__ == "__main__":
    os.chdir(ROOT)
    unittest.main()
