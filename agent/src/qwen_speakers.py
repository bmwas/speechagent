"""Qwen3-Omni built-in speakers. No reference WAV — only preset voices."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# Qwen3-Omni checkpoint default when no speaker is specified
DEFAULT_QWEN_SPEAKER = "ethan"
QWEN_SPEAKER_ENV = "QWEN_SPEAKER"


class QwenSpeakerNotFound(ValueError):
    """Unknown Qwen speaker id."""


@dataclass(frozen=True)
class QwenSpeaker:
    id: str
    name: str
    gender: str  # "female" | "male"
    blurb: str
    featured: bool


_QWEN_SPEAKERS: tuple[QwenSpeaker, ...] = (
    QwenSpeaker(
        id="chelsie",
        name="Chelsie",
        gender="female",
        blurb="honeyed warmth with luminous clarity",
        featured=True,
    ),
    QwenSpeaker(
        id="ethan",
        name="Ethan",
        gender="male",
        blurb="bright energy with an approachable vibe",
        featured=True,
    ),
    QwenSpeaker(
        id="aiden",
        name="Aiden",
        gender="male",
        blurb="warm laid-back American charm",
        featured=True,
    ),
)

_BY_ID = {s.id: s for s in _QWEN_SPEAKERS}


def list_qwen_speakers() -> list[QwenSpeaker]:
    return list(_QWEN_SPEAKERS)


def resolve_qwen_speaker(speaker_id: str | None) -> QwenSpeaker:
    """Resolve a Qwen speaker id. Falls back to env or checkpoint default."""
    raw = (speaker_id or "").strip().lower()
    if not raw:
        # Use env default, then checkpoint default
        raw = (os.environ.get(QWEN_SPEAKER_ENV) or "").strip().lower()
    if not raw:
        raw = DEFAULT_QWEN_SPEAKER
    if raw not in _BY_ID:
        raise QwenSpeakerNotFound(f"unknown Qwen speaker '{raw}'")
    return _BY_ID[raw]


def qwen_speakers_public() -> list[dict[str, Any]]:
    """Return the public catalog for the frontend."""
    out = []
    for speaker in _QWEN_SPEAKERS:
        out.append(
            {
                "id": speaker.id,
                "name": speaker.name,
                "gender": speaker.gender,
                "blurb": speaker.blurb,
                "featured": speaker.featured,
            }
        )
    return out


def default_qwen_speaker_id() -> str:
    """Return the speaker id that will be used if none is specified."""
    raw = (os.environ.get(QWEN_SPEAKER_ENV) or "").strip().lower()
    if raw and raw in _BY_ID:
        return raw
    return DEFAULT_QWEN_SPEAKER
