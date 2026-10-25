#!/usr/bin/env python3
"""Guard v0.30.0 Qwen2.5-Omni talker pad-remap when prompt_token_ids is None.

The stock line:

    sampling_metadata.prompt_token_ids[sampling_metadata.prompt_token_ids == 152064] = 8448

raises TypeError on /v1/video/chat/stream (prompt_token_ids is None). The
orchestrator then evicts stage-1 for the rest of the process.
"""

from __future__ import annotations

import sys
from pathlib import Path

MARKER = "speechagent_qwen_talker_pad_guard"
NEEDLE = "sampling_metadata.prompt_token_ids[sampling_metadata.prompt_token_ids == 152064] = 8448"


def patch_source(src: str) -> tuple[str, str]:
    if MARKER in src:
        return src, "already"
    if NEEDLE not in src:
        return src, "missing"
    out: list[str] = []
    for line in src.splitlines(keepends=True):
        if NEEDLE in line:
            indent = line[: len(line) - len(line.lstrip(" \t"))]
            nl = "\n" if line.endswith("\n") else ""
            stripped = line.lstrip(" \t")
            out.append(f"{indent}# {MARKER}{nl}")
            out.append(
                f"{indent}if getattr(sampling_metadata, 'prompt_token_ids', None) is not None:{nl}"
            )
            out.append(f"{indent}    {stripped}")
        else:
            out.append(line)
    return "".join(out), "patched"


def _candidate_files() -> list[Path]:
    roots = [
        Path("/usr/local/lib/python3.12/dist-packages"),
        Path("/usr/local/lib/python3.11/dist-packages"),
        Path("/usr/local/lib/python3.12/site-packages"),
        Path("/usr/local/lib/python3.11/site-packages"),
    ]
    found: list[Path] = []
    for root in roots:
        found.extend(root.glob("vllm_omni/model_executor/models/qwen2_5_omni/qwen2_5_omni.py"))
    if found:
        return found
    lib = Path("/usr/local/lib")
    if lib.is_dir():
        return [
            p
            for p in lib.rglob("qwen2_5_omni.py")
            if p.parent.name == "qwen2_5_omni" and "vllm_omni" in p.parts
        ]
    return []


def main() -> int:
    files = _candidate_files()
    if not files:
        print("model-server qwen talker patch FATAL: qwen2_5_omni.py not found")
        return 1
    any_ok = False
    for path in files:
        src = path.read_text(encoding="utf-8")
        new, status = patch_source(src)
        print(f"model-server qwen talker patch {path}: {status}")
        if status == "missing":
            for i, line in enumerate(src.splitlines(), 1):
                if "152064" in line or ("prompt_token_ids" in line and "8448" in line):
                    print(f"  L{i}: {line[:220]}")
            return 1
        if status == "patched":
            path.write_text(new, encoding="utf-8")
            cache = path.parent / "__pycache__"
            if cache.is_dir():
                for pyc in cache.glob("qwen2_5_omni*.pyc"):
                    pyc.unlink()
        any_ok = True
    return 0 if any_ok else 1


def _selftest() -> None:
    sample = (
        "        if sampling_metadata is not None:\n"
        "            # pad remap\n"
        f"            {NEEDLE}\n"
        "        return OmniOutput()\n"
    )
    patched, status = patch_source(sample)
    assert status == "patched", status
    assert MARKER in patched
    assert "getattr(sampling_metadata, 'prompt_token_ids', None) is not None" in patched
    again, status2 = patch_source(patched)
    assert status2 == "already", status2
    assert again == patched
    missing, status3 = patch_source("return None\n")
    assert status3 == "missing", status3
    assert missing == "return None\n"


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        print("model-server qwen talker patch selftest ok")
        raise SystemExit(0)
    raise SystemExit(main())
