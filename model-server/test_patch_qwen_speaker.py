"""v0.30.0 video-stream speaker patch (session.config -> talker)."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PATCH = ROOT / "deploy" / "patch_qwen_speaker.py"


class PatchQwenSpeakerTests(unittest.TestCase):
    def test_selftest(self) -> None:
        out = subprocess.run(
            [sys.executable, str(PATCH), "--selftest"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(out.returncode, 0, msg=out.stderr or out.stdout)
        self.assertIn("selftest ok", out.stdout)


if __name__ == "__main__":
    unittest.main()
