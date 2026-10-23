"""FastAPI token + health. Starts the LiveKit bot on first Connect."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent import SpeechSession
from tokens import livekit_url, token_for_user
from trace import trace, trace_exc, trace_stream
from voices import (
    DEFAULT_VOICE_ID,
    VoiceNotFound,
    catalog_public,
    resolve_voice,
    validate_catalog_voice,
    voice_conflict,
)

LOG = logging.getLogger("token_server")

LIVEKIT_PUBLIC_URL = (os.environ.get("LIVEKIT_PUBLIC_URL") or "").strip()
MSE_LAN_IP = (os.environ.get("MSE_LAN_IP") or "").strip()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_bots: dict[str, SpeechSession] = {}
_bots_lock: asyncio.Lock | None = None
_humans: dict[str, set[str]] = {}


def _livekit_host_port() -> tuple[str, int]:
    raw = livekit_url()
    if "://" not in raw:
        raw = "ws://" + raw
    from urllib.parse import urlparse

    parsed = urlparse(raw)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8880
    return host, port


async def livekit_tcp_ok() -> tuple[bool, str]:
    host, port = _livekit_host_port()
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), 2.0)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return True, f"{host}:{port}"
    except Exception as exc:
        return False, f"{host}:{port} {exc}"


def public_ws_url(request: Request | None = None) -> str:
    if LIVEKIT_PUBLIC_URL:
        return LIVEKIT_PUBLIC_URL
    if request is None:
        return ""
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    host = host.split(",")[0].strip()
    proto = (request.headers.get("x-forwarded-proto") or request.url.scheme or "https").split(",")[0].strip()
    ws = "wss" if proto == "https" else "ws"
    if not host:
        return ""
    return f"{ws}://{host}"


@app.get("/health")
async def health():
    brains = sum(1 for b in _bots.values() if b._brain_started and b._running)
    lk_ok, lk_detail = await livekit_tcp_ok()
    voices = {
        rid: getattr(bot, "voice_id", DEFAULT_VOICE_ID)
        for rid, bot in _bots.items()
        if getattr(bot, "_running", False)
    }
    return {
        "ok": True,
        "model": os.environ.get("MODEL_NAME", ""),
        "transport": "livekit",
        "bots": sum(1 for b in _bots.values() if b._running),
        "brains": brains,
        "rooms": list(_bots.keys()),
        "voices": voices,
        "livekit_url": os.environ.get("LIVEKIT_URL", ""),
        "livekit_public_url": LIVEKIT_PUBLIC_URL or "(same-origin wss /rtc)",
        "livekit_tcp": lk_ok,
        "livekit_tcp_detail": lk_detail,
    }


@app.get("/api/voices")
async def api_voices():
    return {"ok": True, "default": DEFAULT_VOICE_ID, "voices": catalog_public()}


@app.post("/api/livekit/token")
async def livekit_token(request: Request):
    global _bots_lock
    if _bots_lock is None:
        _bots_lock = asyncio.Lock()

    role = (request.query_params.get("role") or "patient").strip().lower() or "patient"
    room_id = (request.query_params.get("room") or "default").strip() or "default"
    identity = f"{role}-{uuid.uuid4().hex[:8]}"
    try:
        spec = resolve_voice(request.query_params.get("voice"))
        validate_catalog_voice(spec)
    except VoiceNotFound as exc:
        trace_stream("speech", "voice.validate_fail", error=str(exc)[:200], room=room_id)
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    except ValueError as exc:
        trace_stream(
            "speech",
            "voice.validate_fail",
            voice_id=getattr(spec, "id", ""),
            error=str(exc)[:200],
            room=room_id,
        )
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    lk_ok, lk_detail = await livekit_tcp_ok()
    if not lk_ok:
        trace_stream("speech", "error.livekit_down", detail=lk_detail)
        return JSONResponse(
            {
                "ok": False,
                "error": f"LiveKit SFU is not listening ({lk_detail}). Recreate the livekit service.",
            },
            status_code=503,
        )
    try:
        token = token_for_user(room_id, identity, name=role)
    except Exception as exc:
        trace_exc("speech", "error.token", exc)
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=503)

    async with _bots_lock:
        bot = _bots.get(room_id)
        conflict = voice_conflict(bot, spec.id)
        if conflict:
            trace_stream(
                "speech",
                "voice.conflict",
                room=room_id,
                requested=spec.id,
                active=conflict,
            )
            return JSONResponse(
                {
                    "ok": False,
                    "error": (
                        f"Voice '{conflict}' is already speaking in this room. "
                        "Disconnect first, then pick a new voice and Connect again."
                    ),
                    "voice_id": conflict,
                    "requested": spec.id,
                },
                status_code=409,
            )
        _humans.setdefault(room_id, set()).add(identity)
        if bot is None or not bot._running:
            bot = SpeechSession(
                room_id,
                on_stopped=lambda rid=room_id: _bots.pop(rid, None),
                voice_id=spec.id,
            )
            _bots[room_id] = bot
            try:
                await asyncio.wait_for(bot.connect_and_publish(), timeout=8.0)
            except Exception as exc:
                trace_exc("speech", "error.bot_connect", exc, room=room_id)
                try:
                    await bot.stop()
                except Exception:
                    pass
                _bots.pop(room_id, None)
                humans = _humans.get(room_id)
                if humans is not None:
                    humans.discard(identity)
                return JSONResponse({"ok": False, "error": f"agent bot failed: {exc}"}, status_code=503)
            bot.spawn_brain()
            trace_stream(
                "speech",
                "voice.accepted",
                room=room_id,
                voice_id=spec.id,
                identity=identity,
            )

    url = public_ws_url(request)
    trace_stream(
        "speech",
        "livekit.token",
        role=role,
        room=room_id,
        identity=identity,
        url=url,
        sfu=lk_detail,
        voice_id=spec.id,
    )
    trace("speech", "session.open", role=role, room=room_id, identity=identity, voice_id=spec.id)
    return {
        "ok": True,
        "url": url,
        "token": token,
        "room": room_id,
        "identity": identity,
        "role": role,
        "model": os.environ.get("MODEL_NAME", ""),
        "voice_id": spec.id,
    }


@app.post("/api/livekit/leave")
async def livekit_leave(request: Request):
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        body = {}
    room_id = str(body.get("room") or request.query_params.get("room") or "default")
    identity = str(body.get("identity") or "")
    humans = _humans.setdefault(room_id, set())
    if identity:
        humans.discard(identity)
    bot = _bots.get(room_id)
    if bot is not None and not humans:
        await bot.stop()
        _bots.pop(room_id, None)
    return {"ok": True}


def run() -> None:
    port = int(os.environ.get("PORT") or os.environ.get("AGENT_PORT") or "8860")
    LOG.info("token server :%s model=%s", port, os.environ.get("MODEL_NAME", ""))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
