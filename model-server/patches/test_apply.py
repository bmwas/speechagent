#!/usr/bin/env python3
"""Version/anchor assertions and dry-run of the v0.30.0 require-response patch."""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import apply as apply_mod


UPSTREAM = Path("/tmp/vllm-omni-v030/vllm_omni")


class ApplyPatchTests(unittest.TestCase):
    def test_anchors_are_unique_in_upstream_tree(self) -> None:
        if not UPSTREAM.is_dir():
            self.skipTest("upstream v0.30.0 checkout is not present")
        files = {
            apply_mod.OMNI_IMPORT_ANCHOR: UPSTREAM / "model_executor/models/minicpmo_4_5/minicpmo_4_5_omni.py",
            apply_mod.OMNI_PREPARE_ANCHOR: UPSTREAM / "model_executor/models/minicpmo_4_5/minicpmo_4_5_omni.py",
            apply_mod.OMNI_FINALIZE_ANCHOR: UPSTREAM / "model_executor/models/minicpmo_4_5/minicpmo_4_5_omni.py",
            apply_mod.OMNI_FINISHED_ANCHOR: UPSTREAM / "model_executor/models/minicpmo_4_5/minicpmo_4_5_omni.py",
            apply_mod.OMNI_PREPARE_DEF: UPSTREAM / "model_executor/models/minicpmo_4_5/minicpmo_4_5_omni.py",
            apply_mod.RUNNER_ANCHOR: UPSTREAM / "engine/duplex/session/runner.py",
            apply_mod.INPUT_INIT_ANCHOR: UPSTREAM / "model_executor/models/minicpmo_4_5/duplex/input.py",
            apply_mod.INPUT_CLEAR_ANCHOR: UPSTREAM / "model_executor/models/minicpmo_4_5/duplex/input.py",
            apply_mod.INPUT_SPEECH_ANCHOR: UPSTREAM / "model_executor/models/minicpmo_4_5/duplex/input.py",
            apply_mod.INPUT_OUT_ANCHOR: UPSTREAM / "model_executor/models/minicpmo_4_5/duplex/input.py",
            apply_mod.INPUT_COMMIT_ANCHOR: UPSTREAM / "model_executor/models/minicpmo_4_5/duplex/input.py",
            apply_mod.INPUT_COMMIT_RESET_ANCHOR: UPSTREAM / "model_executor/models/minicpmo_4_5/duplex/input.py",
            apply_mod.INPUT_FLUSH_ANCHOR: UPSTREAM / "model_executor/models/minicpmo_4_5/duplex/input.py",
        }
        for anchor, path in files.items():
            text = path.read_text(encoding="utf-8")
            self.assertEqual(text.count(anchor), 1, f"{path} {anchor!r}")

    def test_apply_is_idempotent_and_fail_closed(self) -> None:
        if not UPSTREAM.is_dir():
            self.skipTest("upstream v0.30.0 checkout is not present")
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "vllm_omni"
            shutil.copytree(UPSTREAM, dest)
            policy = Path(__file__).resolve().parent / "require_response_policy.py"
            first = apply_mod.apply(omni_root=dest, policy_src=policy)
            second = apply_mod.apply(omni_root=dest, policy_src=policy)
            self.assertTrue((dest / "model_executor/models/minicpmo_4_5/duplex/require_response_policy.py").is_file())
            omni = (dest / "model_executor/models/minicpmo_4_5/minicpmo_4_5_omni.py").read_text(encoding="utf-8")
            runner = (dest / "engine/duplex/session/runner.py").read_text(encoding="utf-8")
            inp = (dest / "model_executor/models/minicpmo_4_5/duplex/input.py").read_text(encoding="utf-8")
            self.assertIn("override_listen", omni)
            self.assertIn('payload["response_required"] = True', runner)
            self.assertIn('out["response_required"] = True', inp)
            self.assertNotIn("response.create retry", omni)
            for value in second.values():
                if value in {"copied", "0.30.0"}:
                    continue
                self.assertEqual(value, "already", second)
            self.assertIn("applied", first.values())

    def test_wrong_version_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "vllm_omni"
            dest.mkdir()
            with self.assertRaises(apply_mod.PatchError) as ctx:
                apply_mod.apply(omni_root=dest, installed_version="0.29.0")
            self.assertIn("0.29.0", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
