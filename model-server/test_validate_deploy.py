"""Deploy YAML validation (prevents vLLM orchestrator token-limit crashes)."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEPLOY = ROOT / "deploy"
VALIDATOR = DEPLOY / "validate_deploy.py"


class ValidateDeployTests(unittest.TestCase):
    def test_qwen3_stage_limits(self) -> None:
        out = subprocess.run(
            [sys.executable, str(VALIDATOR), str(DEPLOY / "qwen3_omni.yaml")],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(out.returncode, 0, msg=out.stderr or out.stdout)
        self.assertIn("stage 2:", out.stdout)
        self.assertIn("max_model_len=32768", out.stdout)

    def test_rejects_missing_max_model_len(self) -> None:
        bad = """
stages:
  - stage_id: 2
    max_num_batched_tokens: 32768
    default_sampling_params:
      max_tokens: 32768
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            fh.write(bad)
            path = fh.name
        try:
            out = subprocess.run(
                [sys.executable, str(VALIDATOR), path],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(out.returncode, 0)
            self.assertIn("max_model_len unset", out.stderr)
        finally:
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
