#!/usr/bin/env python3
"""Forward session.config speaker into the Qwen3-Omni talker on v0.30.0.

Stock vllm-omni v0.30.0 StreamingVideoSessionConfig has no speaker field.
Pydantic drops the client's value, ChatCompletionRequest is built without
one, and the talker falls back to the first talker_config.speaker_id key
(Chelsie). Chat completions already copy request.speaker onto
engine_prompt["additional_information"]["speaker"]; the video-stream path
never set that field.

This patch:
  1. Accepts speaker on session.config.
  2. Writes additional_information.speaker after preprocess so
     extract_speaker_from_request can reach the talker.
"""

from __future__ import annotations

import sys
from pathlib import Path

MARKER = "speechagent_qwen_speaker"
CONFIG_NEEDLE = 'description="Custom system prompt."'
PREPROCESS_NEEDLE = "engine_prompt = await self._preprocess_to_engine_prompt(chat_request)"


def _speaker_field(indent: str, nl: str) -> str:
    inner = indent + "    "
    return (
        f"{indent}# {MARKER}{nl}"
        f"{indent}speaker: str | None = Field({nl}"
        f"{inner}default=None,{nl}"
        f'{inner}description="Qwen3-Omni built-in speaker (chelsie, ethan, aiden).",{nl}'
        f"{indent}){nl}"
    )


def _inject_block(indent: str, nl: str) -> str:
    inner = indent + "    "
    return (
        f"{indent}# {MARKER}{nl}"
        f"{indent}_speaker = getattr(config, 'speaker', None){nl}"
        f"{indent}if isinstance(_speaker, str) and _speaker.strip():{nl}"
        f"{inner}_name = _speaker.strip().lower(){nl}"
        f"{inner}if isinstance(engine_prompt, dict):{nl}"
        f"{inner}    _info = engine_prompt.get('additional_information'){nl}"
        f"{inner}    if not isinstance(_info, dict):{nl}"
        f"{inner}        _info = {{}}{nl}"
        f"{inner}        engine_prompt['additional_information'] = _info{nl}"
        f"{inner}    _info['speaker'] = [_name]{nl}"
        f"{inner}else:{nl}"
        f"{inner}    _info = getattr(engine_prompt, 'additional_information', None){nl}"
        f"{inner}    if not isinstance(_info, dict):{nl}"
        f"{inner}        _info = {{}}{nl}"
        f"{inner}        try:{nl}"
        f"{inner}            engine_prompt.additional_information = _info{nl}"
        f"{inner}        except Exception:{nl}"
        f"{inner}            _info = None{nl}"
        f"{inner}    if isinstance(_info, dict):{nl}"
        f"{inner}        _info['speaker'] = [_name]{nl}"
    )


def _insert_after_system_prompt(src: str) -> tuple[str, bool]:
    if f"speaker: str | None = Field(" in src and MARKER in src:
        return src, False
    lines = src.splitlines(keepends=True)
    out: list[str] = []
    inserted = False
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        if (not inserted) and CONFIG_NEEDLE in line:
            # Consume the closing ')' of the Field( call.
            j = i + 1
            while j < len(lines) and lines[j].strip() in (")", "),"):
                out.append(lines[j])
                j += 1
                break
            else:
                i += 1
                continue
            indent = line[: len(line) - len(line.lstrip(" \t"))]
            # Field( is indented one more than the attribute; speaker uses
            # the attribute indent (same as system_prompt).
            attr_indent = indent[: max(0, len(indent) - 4)] if indent.endswith("    ") else indent
            # If system_prompt itself is the attribute line, indent is already
            # the attribute indent when CONFIG_NEEDLE is on the description
            # line (typically 8 spaces). Attribute indent is 4 spaces.
            if "system_prompt" not in line:
                attr_indent = indent[: max(0, len(indent) - 4)]
            nl = "\n" if line.endswith("\n") else ""
            out.append(_speaker_field(attr_indent, nl))
            inserted = True
            i = j
            continue
        i += 1
    return "".join(out), inserted


def _insert_after_preprocess(src: str) -> tuple[str, bool]:
    if MARKER in src and "_speaker = getattr(config, 'speaker', None)" in src:
        return src, False
    if PREPROCESS_NEEDLE not in src:
        return src, False
    lines = src.splitlines(keepends=True)
    out: list[str] = []
    inserted = False
    for line in lines:
        out.append(line)
        if (not inserted) and PREPROCESS_NEEDLE in line:
            indent = line[: len(line) - len(line.lstrip(" \t"))]
            nl = "\n" if line.endswith("\n") else ""
            out.append(_inject_block(indent, nl))
            inserted = True
    return "".join(out), inserted


def patch_source(src: str) -> tuple[str, str]:
    if MARKER in src and "speaker: str | None = Field(" in src and "_speaker = getattr(config" in src:
        return src, "already"
    new, did_config = _insert_after_system_prompt(src)
    new, did_inject = _insert_after_preprocess(new)
    if MARKER in new and "speaker: str | None = Field(" in new and "_speaker = getattr(config" in new:
        return new, "patched" if (did_config or did_inject or new != src) else "already"
    return src, "missing"


def _candidate_files() -> list[Path]:
    roots = [
        Path("/usr/local/lib/python3.12/dist-packages"),
        Path("/usr/local/lib/python3.11/dist-packages"),
        Path("/usr/local/lib/python3.12/site-packages"),
        Path("/usr/local/lib/python3.11/site-packages"),
    ]
    found: list[Path] = []
    for root in roots:
        found.extend(root.glob("vllm_omni/entrypoints/openai/video_stream_base.py"))
    if found:
        return found
    lib = Path("/usr/local/lib")
    if lib.is_dir():
        return [
            p
            for p in lib.rglob("video_stream_base.py")
            if "vllm_omni" in p.parts and p.parent.name == "openai"
        ]
    return []


def main() -> int:
    files = _candidate_files()
    if not files:
        print("model-server qwen speaker patch FATAL: video_stream_base.py not found")
        return 1
    any_ok = False
    for path in files:
        src = path.read_text(encoding="utf-8")
        new, status = patch_source(src)
        print(f"model-server qwen speaker patch {path}: {status}")
        if status == "missing":
            for i, line in enumerate(src.splitlines(), 1):
                if "system_prompt" in line or "preprocess_to_engine_prompt" in line:
                    print(f"  L{i}: {line[:220]}")
            return 1
        if status == "patched":
            path.write_text(new, encoding="utf-8")
            cache = path.parent / "__pycache__"
            if cache.is_dir():
                for pyc in cache.glob("video_stream_base*.pyc"):
                    pyc.unlink()
        any_ok = True
    return 0 if any_ok else 1


def _sample_src() -> str:
    return (
        "class StreamingVideoSessionConfig(BaseModel):\n"
        '    system_prompt: str | None = Field(\n'
        "        default=None,\n"
        '        description="Custom system prompt.",\n'
        "    )\n"
        "    use_audio_in_video: bool = Field(\n"
        "        default=True,\n"
        "    )\n"
        "\n"
        "    async def _process_query_engine(self):\n"
        "        engine_prompt = await self._preprocess_to_engine_prompt(chat_request)\n"
        "        decoded_ready_ts_ms = 0\n"
    )


def _selftest() -> None:
    patched, status = patch_source(_sample_src())
    assert status == "patched", status
    assert MARKER in patched
    assert "speaker: str | None = Field(" in patched
    assert "_speaker = getattr(config, 'speaker', None)" in patched
    assert "_info['speaker'] = [_name]" in patched
    assert 'description="Qwen3-Omni built-in speaker' in patched
    # speaker field sits with the other attributes, not inside system_prompt
    sys_idx = patched.index("system_prompt")
    use_idx = patched.index("use_audio_in_video")
    spk_idx = patched.index("speaker: str | None = Field(")
    assert sys_idx < spk_idx < use_idx
    again, status2 = patch_source(patched)
    assert status2 == "already", status2
    assert again == patched
    missing, status3 = patch_source("return None\n")
    assert status3 == "missing", status3


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        print("model-server qwen speaker patch selftest ok")
        raise SystemExit(0)
    raise SystemExit(main())
