"""FastAPI token + health. Starts the LiveKit bot on first Connect."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent import SpeechSession, is_minicpm
from qwen_omni import probe_model_server, probe_video_stream
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
from qwen_speakers import (
    QwenSpeakerNotFound,
    default_qwen_speaker_id,
    qwen_speakers_public,
    resolve_qwen_speaker,
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
_ready_cache: tuple[float, dict[str, Any]] | None = None
_READY_TTL_S = 0.8


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


def _model_host_port() -> tuple[str, int]:
    raw = (os.environ.get("MODEL_BASE_URL") or "http://127.0.0.1:8199").rstrip("/")
    if "://" not in raw:
        raw = "http://" + raw
    parsed = urlparse(raw)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8199
    return host, port


async def model_server_tcp_ok() -> tuple[bool, str]:
    host, port = _model_host_port()
    try:
        _reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), 2.0)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return True, f"{host}:{port}"
    except Exception as exc:
        return False, f"{host}:{port} {exc}"


def _native_worker_http_url() -> str:
    """MiniCPM worker is :22400. MODEL_BASE_URL often still points at vLLM."""
    base = (os.environ.get("NATIVE_WORKER_URL") or os.environ.get("MODEL_BASE_URL") or "").strip().rstrip("/")
    if not base or ":8199" in base:
        base = "http://127.0.0.1:22400"
    if base.startswith("ws://"):
        base = "http://" + base[len("ws://") :]
    elif base.startswith("wss://"):
        base = "https://" + base[len("wss://") :]
    if base.endswith("/v1/worker/duplex"):
        base = base[: -len("/v1/worker/duplex")]
    return base.rstrip("/")


async def probe_native_worker() -> dict[str, Any]:
    """HTTP /health on the native MiniCPM worker. TCP alone is not model_loaded."""
    base = _native_worker_http_url()
    parsed = urlparse(base if "://" in base else f"http://{base}")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 22400
    out: dict[str, Any] = {
        "tcp_ok": False,
        "http_ok": False,
        "model_loaded": False,
        "detail": "",
        "url": f"{base}/health",
    }
    try:
        _reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), 2.0)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        out["tcp_ok"] = True
    except Exception as exc:
        out["detail"] = f"tcp refused: {exc}"[:200]
        return out

    def _http() -> tuple[bool, bool, str]:
        try:
            with urllib.request.urlopen(out["url"], timeout=2) as resp:
                raw = resp.read(800).decode("utf-8", "replace")
                http_ok = 200 <= int(getattr(resp, "status", 200)) < 300
                loaded = False
                try:
                    body = json.loads(raw)
                    loaded = bool(body.get("model_loaded") or body.get("ready") or body.get("ok"))
                except Exception:
                    loaded = http_ok
                return http_ok, loaded, raw[:160]
        except urllib.error.HTTPError as exc:
            return False, False, f"health HTTP {exc.code}"
        except Exception as exc:
            return False, False, str(exc)[:160]

    try:
        http_ok, loaded, detail = await asyncio.to_thread(_http)
        out["http_ok"] = http_ok
        out["model_loaded"] = loaded
        out["detail"] = detail
    except Exception as exc:
        out["detail"] = f"http probe failed: {exc}"[:200]
    return out


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
    backend = (os.environ.get("MODEL_BACKEND") or "vllm").strip().lower()
    model_tcp_ok = True
    model_tcp_detail = "(native worker — not vLLM :8199)"
    if backend == "vllm":
        model_tcp_ok, model_tcp_detail = await model_server_tcp_ok()
    engine_ready = any(
        getattr(bot, "_brain_started", False)
        and getattr(bot, "_running", False)
        and bool(getattr(getattr(bot, "engine", None), "ready", False))
        for bot in _bots.values()
    )
    ok = lk_ok and (model_tcp_ok or backend != "vllm")
    if backend == "vllm" and not model_tcp_ok:
        trace_stream(
            "speech",
            "error.model_down",
            detail=model_tcp_detail,
            model=os.environ.get("MODEL_NAME", ""),
            model_base_url=os.environ.get("MODEL_BASE_URL", ""),
        )
    voices = {
        rid: getattr(bot, "voice_id", DEFAULT_VOICE_ID)
        for rid, bot in _bots.items()
        if getattr(bot, "_running", False)
    }
    # Cache-only — never probe the video socket from /health. Docker's
    # healthcheck timeout is 3s; a load-time WS wait would restart the agent.
    cached = _ready_cache[1] if _ready_cache else {}
    return {
        "ok": ok,
        "model": os.environ.get("MODEL_NAME", ""),
        "model_backend": backend,
        "model_base_url": os.environ.get("MODEL_BASE_URL", ""),
        "transport": "livekit",
        "bots": sum(1 for b in _bots.values() if b._running),
        "brains": brains,
        "engine_ready": engine_ready,
        "admit": bool(cached.get("admit")) if cached else False,
        "phase": cached.get("phase") or ("up" if ok else "down"),
        "stream_ok": bool(cached.get("stream_ok")) if cached else False,
        "rooms": list(_bots.keys()),
        "voices": voices,
        "livekit_url": os.environ.get("LIVEKIT_URL", ""),
        "livekit_public_url": LIVEKIT_PUBLIC_URL or "(same-origin wss /rtc)",
        "livekit_tcp": lk_ok,
        "livekit_tcp_detail": lk_detail,
        "model_tcp": model_tcp_ok,
        "model_tcp_detail": model_tcp_detail,
    }


def _model_family() -> str:
    """Return 'minicpm' or 'qwen' based on MODEL_NAME."""
    return "minicpm" if is_minicpm() else "qwen"


async def admission_status(room_id: str = "default") -> dict[str, Any]:
    """Whether a patient may enter. Model HTTP up is not enough — the live
    speech socket must accept, or an already-running engine must be ready."""
    global _ready_cache
    now = time.monotonic()
    cached = _ready_cache
    if cached and (now - cached[0]) < _READY_TTL_S:
        snap = dict(cached[1])
        snap["room"] = room_id
        return snap

    backend = (os.environ.get("MODEL_BACKEND") or "vllm").strip().lower()
    family = _model_family()
    lk_ok, lk_detail = await livekit_tcp_ok()
    bot = _bots.get(room_id)
    engine_ready = bool(
        bot is not None
        and getattr(bot, "_running", False)
        and bool(getattr(getattr(bot, "engine", None), "ready", False))
    )

    model_tcp = False
    model_http = False
    models_ok = False
    stream_ok = False
    model_detail = ""
    stream_detail = ""

    if backend == "native":
        native = await probe_native_worker()
        model_tcp = bool(native.get("tcp_ok"))
        model_http = bool(native.get("http_ok"))
        models_ok = bool(native.get("model_loaded"))
        model_detail = str(native.get("detail") or "")
        stream_ok = models_ok or engine_ready
        stream_detail = "worker model_loaded" if models_ok else model_detail
    else:
        probe = await probe_model_server()
        model_tcp = bool(probe.get("tcp_ok"))
        model_http = bool(probe.get("http_ok"))
        models_ok = bool(probe.get("models_ok"))
        model_detail = str(probe.get("detail") or "")
        if model_http and models_ok:
            stream = await probe_video_stream()
            stream_ok = bool(stream.get("stream_ok"))
            stream_detail = str(stream.get("stream_detail") or "")
        elif engine_ready:
            stream_ok = True
            stream_detail = "engine session already live"
        else:
            stream_detail = "model HTTP not serving yet"

    model_serving = model_http and models_ok
    admit = bool(lk_ok and model_serving and (stream_ok or engine_ready))
    if not lk_ok:
        phase = "waiting_sfu"
        message = f"LiveKit is not listening ({lk_detail})."
    elif not model_tcp:
        phase = "waiting_model"
        message = "Model server is not listening yet. Weights are still coming up."
    elif not model_serving:
        phase = "waiting_model"
        message = "Model HTTP is up but stages are still loading. Stay in the greenroom."
    elif not (stream_ok or engine_ready):
        phase = "waiting_engine"
        message = "Live speech socket is not accepting yet. Almost there."
    else:
        phase = "ready"
        message = "Engine is ready."

    snap = {
        "ok": True,
        "admit": admit,
        "phase": phase,
        "message": message,
        "livekit_ok": lk_ok,
        "livekit_detail": lk_detail,
        "model_tcp": model_tcp,
        "model_http": model_http,
        "models_ok": models_ok,
        "model_detail": model_detail[:200],
        "stream_ok": stream_ok,
        "stream_detail": stream_detail[:200],
        "engine_ready": engine_ready,
        "model_serving": model_serving,
        "model": os.environ.get("MODEL_NAME", ""),
        "model_backend": backend,
        "family": family,
        "room": room_id,
    }
    _ready_cache = (now, snap)
    return dict(snap)


@app.get("/api/ready")
async def api_ready(request: Request):
    room_id = (request.query_params.get("room") or "default").strip() or "default"
    return await admission_status(room_id)


@app.get("/api/voices")
async def api_voices():
    family = _model_family()
    if family == "minicpm":
        return {
            "ok": True,
            "family": "minicpm",
            "default": DEFAULT_VOICE_ID,
            "voices": catalog_public(),
        }
    else:
        return {
            "ok": True,
            "family": "qwen",
            "default": default_qwen_speaker_id(),
            "voices": qwen_speakers_public(),
        }


@app.post("/api/livekit/token")
async def livekit_token(request: Request):
    global _bots_lock
    if _bots_lock is None:
        _bots_lock = asyncio.Lock()

    role = (request.query_params.get("role") or "patient").strip().lower() or "patient"
    room_id = (request.query_params.get("room") or "default").strip() or "default"
    identity = f"{role}-{uuid.uuid4().hex[:8]}"
    family = _model_family()
    voice_id: str
    if family == "minicpm":
        # MiniCPM: validate against WAV catalog
        try:
            spec = resolve_voice(request.query_params.get("voice"))
            validate_catalog_voice(spec)
            voice_id = spec.id
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
    else:
        # Qwen: validate against built-in speakers (no WAV)
        try:
            qwen_spec = resolve_qwen_speaker(request.query_params.get("voice"))
            voice_id = qwen_spec.id
        except QwenSpeakerNotFound as exc:
            trace_stream("speech", "voice.validate_fail", error=str(exc)[:200], room=room_id)
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
    admission = await admission_status(room_id)
    if not admission.get("admit"):
        trace_stream(
            "speech",
            "greenroom.hold",
            room=room_id,
            phase=admission.get("phase"),
            message=str(admission.get("message") or "")[:200],
        )
        return JSONResponse(
            {
                "ok": False,
                "error": (
                    f"{admission.get('message') or 'Engine is not ready.'} "
                    "Wait in the greenroom — do not talk yet."
                ),
                "phase": admission.get("phase"),
                "ready": admission,
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
        conflict = voice_conflict(bot, voice_id)
        if conflict:
            trace_stream(
                "speech",
                "voice.conflict",
                room=room_id,
                requested=voice_id,
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
                    "requested": voice_id,
                },
                status_code=409,
            )
        _humans.setdefault(room_id, set()).add(identity)
        if bot is None or not bot._running:
            bot = SpeechSession(
                room_id,
                on_stopped=lambda rid=room_id: _bots.pop(rid, None),
                voice_id=voice_id,
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
                voice_id=voice_id,
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
        voice_id=voice_id,
    )
    trace("speech", "session.open", role=role, room=room_id, identity=identity, voice_id=voice_id)
    return {
        "ok": True,
        "url": url,
        "token": token,
        "room": room_id,
        "identity": identity,
        "role": role,
        "model": os.environ.get("MODEL_NAME", ""),
        "voice_id": voice_id,
        "family": family,
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


@app.get("/api/live")
async def api_live(request: Request):
    """Return live session snapshot for the floor and metrics strip."""
    room_id = request.query_params.get("room") or "default"
    bot = _bots.get(room_id)
    if bot is None or not getattr(bot, "_running", False):
        return {"ok": False, "error": "no active session", "room": room_id}

    # Get engine snapshot
    snap: dict[str, Any] = {}
    engine = getattr(bot, "engine", None)
    if engine is not None:
        getter = getattr(engine, "stream_snapshot", None)
        if callable(getter):
            try:
                snap = getter() or {}
            except Exception:
                snap = {}

    # Basic session info
    family = _model_family()
    minicpm = is_minicpm()
    backend = (os.environ.get("MODEL_BACKEND") or "vllm").strip().lower()

    # Floor state
    vad = getattr(bot, "vad", None)
    playout = getattr(bot, "playout", None)
    user_speaking = bool(getattr(vad, "speaking", False)) if vad else False
    agent_audible = bool(getattr(playout, "audible", False)) if playout else False
    engine_generating = bool(getattr(engine, "generating", False)) if engine else False
    engine_ready = bool(getattr(engine, "ready", False)) if engine else False

    # Determine floor holder and agent state
    if agent_audible:
        agent_state = "speaking"
        floor = "agent"
    elif engine_generating:
        agent_state = "thinking"
        floor = "agent"
    elif engine_ready:
        agent_state = "listening"
        floor = "user" if user_speaking else "none"
    else:
        agent_state = "connecting"
        floor = "none"

    # Session score for session lab
    session_score: dict[str, Any] = {}
    session_score_fn = getattr(bot, "session_score_snapshot", None)
    if callable(session_score_fn):
        try:
            session_score = session_score_fn() or {}
        except Exception:
            session_score = {}

    return {
        "ok": True,
        "room": room_id,
        "family": family,
        "backend": backend,
        "model": os.environ.get("MODEL_NAME", ""),
        "voice_id": getattr(bot, "voice_id", ""),
        # Floor
        "floor": floor,
        "user_speaking": user_speaking,
        "agent_state": agent_state,
        "agent_audible": agent_audible,
        # Metrics from engine snapshot
        "ready": snap.get("ready", False),
        "speaking": snap.get("speaking", False),
        "out_q": snap.get("out_q", 0),
        # Qwen-specific
        "frame_n": snap.get("frame_n"),
        "utt_ms": snap.get("utt_ms"),
        "delta_age_ms": snap.get("delta_age_ms"),
        "speaker": snap.get("speaker"),
        # MiniCPM vLLM-specific
        "turn_waiting": snap.get("turn_waiting"),
        "rtf": snap.get("rtf"),
        "buf_ms": snap.get("buf_ms"),
        # Native MiniCPM-specific
        "kv": snap.get("kv"),
        "cost_all_ms_p50": snap.get("cost_all_ms_p50"),
        # Common
        "ring_ms": round(getattr(playout, "ring_ms", 0), 1) if playout else 0,
        "mic_age_ms": snap.get("mic_age_ms"),
        # Turn metrics for session lab (from last completed turn)
        "turn": {
            "ttfa_ms": snap.get("ttfa_ms", -1),
            "ttfa_anchor": snap.get("ttfa_anchor", ""),
            "ttft_ms": snap.get("ttft_ms", -1),
            "rtf": snap.get("rtf", -1),
            "gap_ms": snap.get("gap_ms", -1),
            "turn_audio_ms": snap.get("turn_audio_ms", 0),
        },
        # Session score (aggregates over all turns)
        "session": session_score,
    }


def run() -> None:
    port = int(os.environ.get("PORT") or os.environ.get("AGENT_PORT") or "8860")
    LOG.info("token server :%s model=%s", port, os.environ.get("MODEL_NAME", ""))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
