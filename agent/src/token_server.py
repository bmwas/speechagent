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
from tokens import token_for_user
from trace import trace, trace_exc

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
    return {
        "ok": True,
        "model": os.environ.get("MODEL_NAME", ""),
        "transport": "livekit",
        "bots": sum(1 for b in _bots.values() if b._running),
        "brains": brains,
        "rooms": list(_bots.keys()),
        "livekit_url": os.environ.get("LIVEKIT_URL", ""),
        "livekit_public_url": LIVEKIT_PUBLIC_URL or "(same-origin wss /rtc)",
    }


@app.post("/api/livekit/token")
async def livekit_token(request: Request):
    global _bots_lock
    if _bots_lock is None:
        _bots_lock = asyncio.Lock()

    role = (request.query_params.get("role") or "patient").strip().lower() or "patient"
    room_id = (request.query_params.get("room") or "default").strip() or "default"
    identity = f"{role}-{uuid.uuid4().hex[:8]}"
    try:
        token = token_for_user(room_id, identity, name=role)
    except Exception as exc:
        trace_exc("speech", "error.token", exc)
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=503)

    _humans.setdefault(room_id, set()).add(identity)

    async with _bots_lock:
        bot = _bots.get(room_id)
        if bot is None or not bot._running:
            bot = SpeechSession(room_id, on_stopped=lambda rid=room_id: _bots.pop(rid, None))
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
                return JSONResponse({"ok": False, "error": f"agent bot failed: {exc}"}, status_code=503)
            bot.spawn_brain()

    url = public_ws_url(request)
    trace("speech", "session.open", role=role, room=room_id, identity=identity)
    return {
        "ok": True,
        "url": url,
        "token": token,
        "room": room_id,
        "identity": identity,
        "role": role,
        "model": os.environ.get("MODEL_NAME", ""),
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
    port = int(os.environ.get("PORT", "7860"))
    LOG.info("token server :%s model=%s", port, os.environ.get("MODEL_NAME", ""))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
