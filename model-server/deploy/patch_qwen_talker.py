#!/usr/bin/env python3
"""Guard v0.30.0 Qwen2.5-Omni talker pad-remap when prompt_token_ids is None.

The stock line:

    sampling_metadata.prompt_token_ids[sampling_metadata.prompt_token_ids == 152064] = 8448

raises TypeError on /v1/video/chat/stream (prompt_token_ids is None). The
orchestrator then evicts stage-1 for the rest of the process.

Skipping the remap (the first guard) lets the talker finish but it then
emits only START/PAD/END codec tokens. code2wav strips those, ships an
empty payload, and the consumer wait gate hangs — text arrives, no PCM.

Fill prompt_token_ids from talker additional_information (thinker ids) or
input_ids, convert to a tensor, then remap 152064 → 8448.
"""

from __future__ import annotations

import sys
from pathlib import Path

MARKER = "speechagent_qwen_talker_pad_guard"
NEEDLE = "sampling_metadata.prompt_token_ids[sampling_metadata.prompt_token_ids == 152064] = 8448"


def _guard_block(indent: str, nl: str) -> str:
    # additional_information / input_ids / torch are in scope in talker forward().
    return (
        f"{indent}# {MARKER}{nl}"
        f"{indent}_ptids = getattr(sampling_metadata, 'prompt_token_ids', None){nl}"
        f"{indent}if _ptids is None:{nl}"
        f"{indent}    _src = None{nl}"
        f"{indent}    if additional_information is not None:{nl}"
        f"{indent}        _src = additional_information.get('prompt_token_ids'){nl}"
        f"{indent}    if _src is None:{nl}"
        f"{indent}        _src = input_ids{nl}"
        f"{indent}    _ptids = _src{nl}"
        f"{indent}if _ptids is not None:{nl}"
        f"{indent}    if not isinstance(_ptids, torch.Tensor):{nl}"
        f"{indent}        _ptids = torch.as_tensor(_ptids){nl}"
        f"{indent}    sampling_metadata.prompt_token_ids = _ptids{nl}"
        f"{indent}    sampling_metadata.prompt_token_ids["
        f"sampling_metadata.prompt_token_ids == 152064] = 8448{nl}"
    )


def patch_source(src: str) -> tuple[str, str]:
    if MARKER in src and "additional_information.get('prompt_token_ids')" in src:
        return src, "already"
    if NEEDLE not in src and MARKER not in src:
        return src, "missing"
    out: list[str] = []
    skip_old_guard = False
    for line in src.splitlines(keepends=True):
        stripped = line.lstrip(" \t")
        # Replace a previous skip-only guard (MARKER + if-not-None wrap).
        if MARKER in line and "additional_information.get" not in line:
            skip_old_guard = True
            continue
        if skip_old_guard:
            if NEEDLE in line or stripped.startswith("if getattr(sampling_metadata"):
                if NEEDLE in line:
                    indent = line[: len(line) - len(line.lstrip(" \t"))]
                    nl = "\n" if line.endswith("\n") else ""
                    out.append(_guard_block(indent, nl))
                    skip_old_guard = False
                continue
            skip_old_guard = False
        if NEEDLE in line and MARKER not in "".join(out[-3:]):
            indent = line[: len(line) - len(line.lstrip(" \t"))]
            nl = "\n" if line.endswith("\n") else ""
            out.append(_guard_block(indent, nl))
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
    assert "additional_information.get('prompt_token_ids')" in patched
    assert "torch.as_tensor" in patched
    assert NEEDLE.split("[")[0] in patched
    again, status2 = patch_source(patched)
    assert status2 == "already", status2
    assert again == patched
    missing, status3 = patch_source("return None\n")
    assert status3 == "missing", status3
    assert missing == "return None\n"
    old_guard = (
        "        if sampling_metadata is not None:\n"
        f"            # {MARKER}\n"
        "            if getattr(sampling_metadata, 'prompt_token_ids', None) is not None:\n"
        f"                {NEEDLE}\n"
        "        return OmniOutput()\n"
    )
    upgraded, status4 = patch_source(old_guard)
    assert status4 == "patched", status4
    assert "additional_information.get('prompt_token_ids')" in upgraded
    assert "getattr(sampling_metadata, 'prompt_token_ids', None) is not None" not in upgraded


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        print("model-server qwen talker patch selftest ok")
        raise SystemExit(0)
    raise SystemExit(main())
