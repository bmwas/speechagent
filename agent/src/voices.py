"""Allowlisted MiniCPM reference voices. Resolved at session setup only."""

from __future__ import annotations

import hashlib
import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_VOICE_ID = "original"
MIN_DURATION_S = 3.0
MAX_DURATION_S = 10.0
REQUIRED_RATE = 16000
REQUIRED_CHANNELS = 1
REQUIRED_SAMPWIDTH = 2


class VoiceNotFound(ValueError):
    """Unknown or unsafe voice id."""


@dataclass(frozen=True)
class Voice:
    id: str
    name: str
    gender: str
    blurb: str
    featured: bool
    source: str
    license: str
    filename: str | None


_VOICES: tuple[Voice, ...] = (
    Voice(
        id="original",
        name="Original",
        gender="original",
        blurb="the MiniCPM studio clone — our first voice",
        featured=True,
        source="openbmb/MiniCPM-o-4_5 HT_ref_audio.wav",
        license="Apache-2.0",
        filename=None,
    ),
    Voice(
        id="heart",
        name="Heart",
        gender="female",
        blurb="warm sparkle, like a pocket sun",
        featured=True,
        source="hexgrad/Kokoro-82M af_heart",
        license="Apache-2.0",
        filename="heart.wav",
    ),
    Voice(
        id="bella",
        name="Bella",
        gender="female",
        blurb="bright and bouncy",
        featured=True,
        source="Hugging Face CDN upload (unverified provenance)",
        license="unverified CDN upload",
        filename="bella.wav",
    ),
    Voice(
        id="nicole",
        name="Nicole",
        gender="female",
        blurb="soft evening tea",
        featured=False,
        source="Hugging Face CDN upload (unverified provenance)",
        license="unverified CDN upload",
        filename="nicole.wav",
    ),
    Voice(
        id="michael",
        name="Michael",
        gender="male",
        blurb="calm and clear",
        featured=True,
        source="Hugging Face CDN upload (unverified provenance)",
        license="unverified CDN upload",
        filename="michael.wav",
    ),
    Voice(
        id="ryan",
        name="Ryan",
        gender="male",
        blurb="friendly radio host",
        featured=True,
        source="rhasspy/piper-voices en_US-ryan-high",
        license="CC BY-NC-SA 4.0 (noncommercial)",
        filename="ryan.wav",
    ),
    Voice(
        id="joe",
        name="Joe",
        gender="male",
        blurb="easygoing neighbor",
        featured=False,
        source="rhasspy/piper-voices en_US-joe-medium",
        license="CC0 dataset; Lessac-derived risk",
        filename="joe.wav",
    ),
    Voice(
        id="lessac",
        name="Lessac",
        gender="female",
        blurb="storybook narrator",
        featured=False,
        source="rhasspy/piper-voices en_US-lessac-high",
        license="Blizzard 2013 research-oriented",
        filename="lessac.wav",
    ),
    Voice(
        id="amy",
        name="Amy",
        gender="female",
        blurb="gentle and even",
        featured=False,
        source="rhasspy/piper-voices en_US-amy-medium",
        license="See upstream; Lessac-derived risk",
        filename="amy.wav",
    ),
)
_BY_ID = {v.id: v for v in _VOICES}


def _asset_dirs() -> list[Path]:
    here = Path(__file__).resolve().parent
    return [
        here.parent / "assets" / "voices",
        here / "assets" / "voices",
        Path("/app/assets/voices"),
    ]


def asset_dir() -> Path:
    for path in _asset_dirs():
        if path.is_dir():
            return path
    return _asset_dirs()[0]


def manifest_path() -> Path:
    return asset_dir() / "MANIFEST.json"


def asset_path(voice: Voice) -> Path | None:
    if not voice.filename:
        return None
    name = Path(voice.filename).name
    if name != voice.filename or "/" in voice.filename or "\\" in voice.filename:
        raise VoiceNotFound("unsafe voice filename")
    return asset_dir() / name


def list_voices() -> list[Voice]:
    return list(_VOICES)


def resolve_voice(voice_id: str | None) -> Voice:
    raw = (voice_id or "").strip().lower()
    if not raw:
        return _BY_ID[DEFAULT_VOICE_ID]
    if raw not in _BY_ID:
        raise VoiceNotFound(f"unknown voice '{raw}'")
    return _BY_ID[raw]


def catalog_public() -> list[dict[str, Any]]:
    out = []
    for voice in _VOICES:
        out.append(
            {
                "id": voice.id,
                "name": voice.name,
                "gender": voice.gender,
                "blurb": voice.blurb,
                "featured": voice.featured,
                "source": voice.source,
                "license": voice.license,
            }
        )
    return out


def wav_info(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        rate = handle.getframerate()
        sampwidth = handle.getsampwidth()
        frames = handle.getnframes()
    duration_s = frames / rate if rate else 0.0
    return {
        "channels": channels,
        "rate": rate,
        "sampwidth": sampwidth,
        "frames": frames,
        "duration_s": duration_s,
        "duration_ms": int(duration_s * 1000),
    }


def validate_wav(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing WAV: {path}")
    raw = path.read_bytes()
    if len(raw) < 64 or raw[:4] != b"RIFF":
        raise ValueError(f"not a RIFF WAV: {path.name}")
    info = wav_info(path)
    if info["channels"] != REQUIRED_CHANNELS:
        raise ValueError(f"{path.name} must be mono")
    if info["rate"] != REQUIRED_RATE:
        raise ValueError(f"{path.name} must be {REQUIRED_RATE} Hz")
    if info["sampwidth"] != REQUIRED_SAMPWIDTH:
        raise ValueError(f"{path.name} must be 16-bit PCM")
    if not (MIN_DURATION_S <= info["duration_s"] <= MAX_DURATION_S):
        raise ValueError(
            f"{path.name} duration {info['duration_s']:.3f}s not in "
            f"{MIN_DURATION_S:.0f}-{MAX_DURATION_S:.0f}s"
        )
    info["sha256"] = hashlib.sha256(raw).hexdigest()
    info["bytes"] = len(raw)
    return info


def _manifest_hashes() -> dict[str, str]:
    path = manifest_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, str] = {}
    for row in data.get("voices") or []:
        vid = str(row.get("id") or "")
        sha = str(row.get("normalized_sha256") or "")
        if vid and sha:
            out[vid] = sha
    return out


def validate_catalog_voice(voice: Voice) -> dict[str, Any]:
    """Validate a selectable catalog voice. Original uses bundled HT at handshake."""
    if voice.filename is None:
        return {"voice_id": voice.id, "bundled": True}
    path = asset_path(voice)
    if path is None:
        raise ValueError(f"voice '{voice.id}' has no asset")
    info = validate_wav(path)
    expected = _manifest_hashes().get(voice.id)
    if expected and info["sha256"] != expected:
        raise ValueError(f"voice '{voice.id}' hash mismatch")
    info["voice_id"] = voice.id
    info["path"] = str(path)
    return info


def voice_conflict(bot: Any, requested_id: str) -> str | None:
    if bot is None or not getattr(bot, "_running", False):
        return None
    current = str(getattr(bot, "voice_id", DEFAULT_VOICE_ID) or DEFAULT_VOICE_ID)
    if current != requested_id:
        return current
    return None


def expected_ids() -> Iterable[str]:
    return tuple(_BY_ID)
