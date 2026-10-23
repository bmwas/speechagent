#!/usr/bin/env python3
"""Client turn-tracker and non-regression tests for MiniCPM response-required."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from require_response import TurnResponseTracker, require_response_enabled

ROOT = Path(__file__).resolve().parents[1]
DUPLEX = ROOT / "src" / "duplex_model.py"


class RequireResponseEnabledTests(unittest.TestCase):
    def test_default_on(self) -> None:
        self.assertTrue(require_response_enabled(""))
        self.assertTrue(require_response_enabled("1"))
        self.assertTrue(require_response_enabled("true"))

    def test_kill_switch(self) -> None:
        self.assertFalse(require_response_enabled("0"))
        self.assertFalse(require_response_enabled("false"))
        self.assertFalse(require_response_enabled("off"))


class TurnTrackerTests(unittest.TestCase):
    def test_speech_then_terminal_marks_once(self) -> None:
        t = TurnResponseTracker(enabled=True, unanswered_ms=8000)
        t.on_speech_start(now=1.0)
        self.assertEqual(t.turn_id, 1)
        self.assertEqual(t.state, "speech")
        self.assertEqual(t.marker_fields(is_speech=True), {"turn_id": 1})
        self.assertTrue(t.mark_terminal(now=2.0))
        self.assertFalse(t.mark_terminal(now=2.1))
        self.assertEqual(t.marker_fields(is_speech=True)["response_required"], True)
        t.mark_commit(now=2.2)
        fields = t.marker_fields()
        self.assertEqual(fields["turn_id"], 1)
        self.assertTrue(fields["response_required"])

    def test_exactly_one_response_per_turn(self) -> None:
        t = TurnResponseTracker(enabled=True)
        t.on_speech_start(now=1.0)
        t.mark_terminal(now=2.0)
        t.mark_commit(now=2.1)
        self.assertEqual(t.on_response_created("r1", now=3.0), "")
        self.assertEqual(t.on_response_created("r1", now=3.05), "")
        self.assertEqual(t.on_response_created("r2", now=3.1), "duplicate")
        self.assertFalse(t.waiting())
        self.assertNotIn("response_required", t.marker_fields())

    def test_silence_without_speech_stays_idle(self) -> None:
        t = TurnResponseTracker(enabled=True)
        self.assertFalse(t.mark_terminal(now=1.0))
        self.assertEqual(t.marker_fields(), {})
        self.assertIsNone(t.maybe_unanswered(now=20.0))

    def test_ongoing_speech_is_not_required(self) -> None:
        t = TurnResponseTracker(enabled=True)
        t.on_speech_start(now=1.0)
        self.assertNotIn("response_required", t.marker_fields(is_speech=True))

    def test_cancel_and_reconnect_clear_waiting(self) -> None:
        t = TurnResponseTracker(enabled=True)
        t.on_speech_start(now=1.0)
        t.mark_terminal(now=2.0)
        t.on_cancel("barge_in", now=2.5)
        self.assertEqual(t.state, "cancelled")
        t.reset_session(reason="reconnect")
        self.assertEqual(t.state, "idle")
        self.assertFalse(t.waiting())

    def test_unanswered_deadline_once(self) -> None:
        t = TurnResponseTracker(enabled=True, unanswered_ms=1000)
        t.on_speech_start(now=10.0)
        t.mark_terminal(now=11.0)
        t.mark_commit(now=11.0)
        self.assertIsNone(t.maybe_unanswered(now=11.5))
        first = t.maybe_unanswered(now=12.5)
        self.assertIsNotNone(first)
        self.assertGreaterEqual(first["unanswered_ms"], 1000)
        self.assertIsNone(t.maybe_unanswered(now=13.0))

    def test_disabled_sends_no_markers(self) -> None:
        t = TurnResponseTracker(enabled=False)
        t.on_speech_start(now=1.0)
        t.mark_terminal(now=2.0)
        self.assertEqual(t.marker_fields(is_speech=True), {})
        self.assertFalse(t.waiting())

    def test_marker_bytes_and_first_attach_are_once(self) -> None:
        t = TurnResponseTracker(enabled=True)
        t.on_speech_start(now=1.0)
        t.mark_terminal(now=2.0)
        t.mark_commit(now=2.1)
        first = t.marker_fields()
        self.assertTrue(first.get("response_required"))
        self.assertGreater(t.last_extra_bytes, 0)
        self.assertTrue(t.consume_first_marker())
        self.assertFalse(t.consume_first_marker())
        t.note_attach_us(3)
        t.note_attach_us(9)
        self.assertEqual(t.attach_us_max, 9)
        self.assertEqual(t.last_attach_us, 9)


class AudioPathRegressionTests(unittest.TestCase):
    def test_known_good_playout_constants_unchanged(self) -> None:
        import audio
        import duplex_model

        self.assertEqual(audio.PLAYOUT_MAX_WAIT_MS, 300)
        self.assertEqual(audio.HOLD_SILENCE_S, 0.25)
        self.assertEqual(duplex_model.STALL_CANCEL_MS, 45000)
        yaml_text = (ROOT.parent / "model-server" / "deploy" / "minicpmo_4_5.yaml").read_text(encoding="utf-8")
        self.assertIn("codec_chunk_frames: 24", yaml_text)


class NoResponseCreateTests(unittest.TestCase):
    def test_client_never_sends_response_create(self) -> None:
        source = DUPLEX.read_text(encoding="utf-8")
        tree = ast.parse(source)
        sent_types: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "type"
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                ):
                    sent_types.append(value.value)
        self.assertNotIn("response.create", sent_types)
        self.assertIn("input_audio_buffer.commit", sent_types)
        self.assertIn("input_audio_buffer.append", sent_types)


if __name__ == "__main__":
    unittest.main()
