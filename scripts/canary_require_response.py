#!/usr/bin/env python3
"""Synthetic 3000-turn canary for MiniCPM response-required.

Live GPU conversation replay is optional and is refused unless
CANARY_LIVE=1 and a model-server is already running. This script never
changes playout, barge-in, or vocoder settings.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model-server" / "patches"))
sys.path.insert(0, str(ROOT / "agent" / "src"))

from test_require_response_policy import CanaryTests  # noqa: E402


def main() -> int:
    if os.environ.get("CANARY_LIVE", "").strip() in {"1", "true", "yes"}:
        print(
            "live 3000-turn GPU canary is not auto-run: talk through the UI and "
            "compare Grafana unanswered/duplicate panels. Refusing to start a "
            "second model-server against the shared Ada GPU.",
            file=sys.stderr,
        )
        return 2
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CanaryTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        print("canary ok: 3000 valid turns, 0 unanswered, 0 duplicates")
        return 0
    print("canary FAILED — roll back to working-audio-2026-10-23 or set MINICPM_REQUIRE_RESPONSE=0")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
