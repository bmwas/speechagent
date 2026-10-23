#!/usr/bin/env python3
"""Policy, duplicate-guard, and 3000-turn synthetic canary for require-response."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from require_response_policy import RequireLogGate, decide_require_response, require_response_enabled


def _decision(**overrides):
    base = dict(
        enabled=True,
        sampled_is_listen=True,
        force_listen=False,
        is_speech=False,
        response_required=True,
        pending_speech_context=True,
        current_turn_ended=True,
        pending_speech_response_open=False,
        turn_id=1,
        opened_turn_ids=set(),
        latched_turn_id=1,
    )
    base.update(overrides)
    return decide_require_response(**base)


class PolicyTests(unittest.TestCase):
    def test_kill_switch(self) -> None:
        self.assertTrue(require_response_enabled(None))
        self.assertTrue(require_response_enabled(""))
        self.assertFalse(require_response_enabled("0"))
        self.assertEqual(_decision(enabled=False).reason, "disabled")
        self.assertFalse(_decision(enabled=False).override_listen)

    def test_listen_to_speak_on_valid_turn(self) -> None:
        d = _decision()
        self.assertTrue(d.override_listen)
        self.assertTrue(d.block_force_listen)
        self.assertEqual(d.reason, "listen_overridden")

    def test_natural_speak_passthrough(self) -> None:
        d = _decision(sampled_is_listen=False)
        self.assertFalse(d.override_listen)
        self.assertEqual(d.reason, "natural_speak")
        self.assertTrue(d.block_force_listen)

    def test_silence_without_pending_speech(self) -> None:
        d = _decision(pending_speech_context=False, response_required=True)
        self.assertFalse(d.override_listen)
        self.assertEqual(d.reason, "no_pending_speech")

    def test_ongoing_speech_stays_listen_capable(self) -> None:
        d = _decision(is_speech=True, response_required=False, latched_turn_id=None)
        self.assertEqual(d.reason, "user_speaking")
        self.assertFalse(d.override_listen)
        self.assertFalse(d.block_force_listen)

    def test_already_active_response(self) -> None:
        d = _decision(current_turn_ended=False, pending_speech_response_open=True)
        self.assertEqual(d.reason, "response_active")
        self.assertFalse(d.override_listen)

    def test_duplicate_guard(self) -> None:
        d = _decision(opened_turn_ids={1})
        self.assertEqual(d.reason, "duplicate")
        self.assertFalse(d.override_listen)

    def test_noise_not_required(self) -> None:
        d = _decision(response_required=False, latched_turn_id=None, pending_speech_context=False)
        self.assertEqual(d.reason, "not_required")
        self.assertFalse(d.override_listen)


class RequireLogGateTests(unittest.TestCase):
    def test_hot_path_skips_do_not_emit_immediately(self) -> None:
        gate = RequireLogGate(period_s=60.0)
        immediates = 0
        summaries = 0
        for _ in range(1000):
            offered = gate.offer("user_speaking", 1)
            immediates += int(offered.immediate is not None)
            summaries += int(offered.summary is not None)
        self.assertEqual(immediates, 0)
        self.assertEqual(summaries, 0)

    def test_override_logs_once_per_turn(self) -> None:
        gate = RequireLogGate(period_s=60.0)
        first = gate.offer("listen_overridden", 7, session_id="s")
        second = gate.offer("listen_overridden", 7, session_id="s")
        other = gate.offer("listen_overridden", 8, session_id="s")
        self.assertIsNotNone(first.immediate)
        self.assertIsNone(second.immediate)
        self.assertIsNotNone(other.immediate)

    def test_summary_flushes_counts(self) -> None:
        gate = RequireLogGate(period_s=0.0)
        gate.period_s = 0.0
        gate._t0 = 0.0
        offered = gate.offer("armed", 1)
        self.assertIsNotNone(offered.summary)
        self.assertEqual(offered.summary["counts"]["armed"], 1)


class CanaryTests(unittest.TestCase):
    """3,000 valid turns through the patched policy with a listen-biased sampler.

    Unpatched MiniCPM answered ~13/34 live turns. The patch must convert every
    remaining listen into tts_bos without duplicating a turn that already spoke.
    """

    TURNS = 3000

    def test_3000_valid_turns_zero_unanswered_zero_duplicate(self) -> None:
        rng = random.Random(20261023)
        opened: set[int] = set()
        unanswered = 0
        duplicates = 0
        natural = 0
        overridden = 0
        for turn_id in range(1, self.TURNS + 1):
            sampled_listen = rng.random() < 0.62
            first = decide_require_response(
                enabled=True,
                sampled_is_listen=sampled_listen,
                force_listen=False,
                is_speech=False,
                response_required=True,
                pending_speech_context=True,
                current_turn_ended=True,
                pending_speech_response_open=False,
                turn_id=turn_id,
                opened_turn_ids=opened,
                latched_turn_id=turn_id,
            )
            spoke = first.override_listen or first.reason == "natural_speak"
            if not spoke:
                unanswered += 1
                continue
            if first.reason == "natural_speak":
                natural += 1
            else:
                overridden += 1
            opened.add(turn_id)
            second = decide_require_response(
                enabled=True,
                sampled_is_listen=True,
                force_listen=False,
                is_speech=False,
                response_required=True,
                pending_speech_context=True,
                current_turn_ended=True,
                pending_speech_response_open=False,
                turn_id=turn_id,
                opened_turn_ids=opened,
                latched_turn_id=turn_id,
            )
            if second.override_listen:
                duplicates += 1
            elif second.reason != "duplicate":
                self.fail(f"turn {turn_id} second sample reason={second.reason}")

        self.assertEqual(unanswered, 0)
        self.assertEqual(duplicates, 0)
        self.assertEqual(natural + overridden, self.TURNS)
        self.assertGreater(overridden, 0)
        self.assertGreater(natural, 0)

    def test_control_cases_do_not_force_speech(self) -> None:
        cases = [
            dict(is_speech=True, response_required=False, latched_turn_id=None, pending_speech_context=True),
            dict(pending_speech_context=False, response_required=False, latched_turn_id=None),
            dict(current_turn_ended=False, pending_speech_response_open=True),
            dict(enabled=False),
        ]
        for extra in cases:
            d = _decision(sampled_is_listen=True, **extra)
            self.assertFalse(d.override_listen, extra)


if __name__ == "__main__":
    unittest.main()
