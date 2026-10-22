"""LiveKit JWTs for the browser participant and the in-room bot."""

from __future__ import annotations

import datetime
import os

BOT_IDENTITY = "speech-agent"


def livekit_api_key() -> str:
    return (os.environ.get("LIVEKIT_API_KEY") or "").strip()


def livekit_api_secret() -> str:
    return (os.environ.get("LIVEKIT_API_SECRET") or "").strip()


def livekit_url() -> str:
    return (os.environ.get("LIVEKIT_URL") or "ws://127.0.0.1:8880").strip()


def mint_token(
    *,
    identity: str,
    room: str,
    name: str = "",
    can_publish: bool = True,
    can_subscribe: bool = True,
    can_publish_sources: list[str] | None = None,
    ttl_hours: int = 24,
) -> str:
    from livekit import api

    key = livekit_api_key()
    secret = livekit_api_secret()
    if not key or not secret:
        raise RuntimeError("LIVEKIT_API_KEY / LIVEKIT_API_SECRET are not set")
    grants = api.VideoGrants(
        room_join=True,
        room=room,
        can_publish=can_publish,
        can_subscribe=can_subscribe,
        can_publish_data=True,
    )
    if can_publish_sources is not None:
        grants.can_publish_sources = list(can_publish_sources)
    token = (
        api.AccessToken(key, secret)
        .with_identity(identity)
        .with_name(name or identity)
        .with_ttl(datetime.timedelta(hours=ttl_hours))
        .with_grants(grants)
    )
    return token.to_jwt()


def token_for_user(room: str, identity: str, name: str = "patient") -> str:
    return mint_token(
        identity=identity,
        room=room,
        name=name or "patient",
        can_publish=True,
        can_subscribe=True,
        can_publish_sources=["camera", "microphone"],
    )


def token_for_bot(room: str) -> str:
    return mint_token(
        identity=BOT_IDENTITY,
        room=room,
        name="assistant",
        can_publish=True,
        can_subscribe=True,
        can_publish_sources=["microphone"],
    )
