#!/usr/bin/env python3
"""Offline check of committed MiniCPM reference-voice WAVs. Not used at runtime."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "agent" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from voices import (  # noqa: E402
    DEFAULT_VOICE_ID,
    asset_path,
    list_voices,
    manifest_path,
    validate_catalog_voice,
)


def main() -> int:
    manifest = json.loads(manifest_path().read_text(encoding="utf-8"))
    rows = {str(row["id"]): row for row in manifest.get("voices") or []}
    errors: list[str] = []
    seen: set[str] = set()
    for voice in list_voices():
        seen.add(voice.id)
        if voice.id == DEFAULT_VOICE_ID:
            if voice.filename is not None:
                errors.append("original must not ship a replacement WAV")
            continue
        try:
            info = validate_catalog_voice(voice)
        except Exception as exc:
            errors.append(str(exc))
            continue
        row = rows.get(voice.id)
        if not row:
            errors.append(f"{voice.id} missing from MANIFEST.json")
            continue
        path = asset_path(voice)
        if path is None or not path.is_file():
            errors.append(f"{voice.id} asset missing")
            continue
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != row.get("normalized_sha256"):
            errors.append(f"{voice.id} sha256 != manifest")
        if sha != info.get("sha256"):
            errors.append(f"{voice.id} runtime hash mismatch")
    extra = set(rows) - seen
    if extra:
        errors.append(f"manifest extras: {sorted(extra)}")
    if errors:
        print("FAIL")
        for item in errors:
            print(" ", item)
        return 1
    print(f"ok {len(seen)} voices, default={DEFAULT_VOICE_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
