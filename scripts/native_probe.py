#!/usr/bin/env python3
"""Smoke the OpenBMB worker duplex protocol before wiring LiveKit.

Connects to ws://127.0.0.1:22400/v1/worker/duplex (no gateway / no queue),
sends session.init immediately, then 1 s float32 chunks from a WAV.

Usage (on the GPU box, after COMPOSE_PROFILES=native is up):

    python3 scripts/native_probe.py --wav /path/to/16k_mono.wav
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import struct
import sys
import time
import wave
from urllib.parse import urlparse

import websockets

INPUT_RATE = 16000
OUTPUT_RATE = 24000


def pcm16_to_f32le(pcm16: bytes) -> bytes:
    n = len(pcm16) // 2
    samples = struct.unpack("<" + "h" * n, pcm16[: n * 2])
    return struct.pack("<" + "f" * n, *[s / 32768.0 for s in samples])


def load_wav_pcm16(path: str, rate: int = INPUT_RATE) -> bytes:
    with wave.open(path, "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError(f"need 16-bit PCM, got sampwidth={handle.getsampwidth()}")
        channels = handle.getnchannels()
        src_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    n = len(frames) // 2
    samples = list(struct.unpack("<" + "h" * n, frames[: n * 2]))
    if channels > 1:
        mono = []
        for i in range(0, len(samples) - channels + 1, channels):
            mono.append(int(sum(samples[i : i + channels]) / channels))
        samples = mono
    if src_rate != rate and src_rate > 0:
        ratio = src_rate / float(rate)
        new_n = max(1, int(round(len(samples) / ratio)))
        resampled = []
        pos = 0.0
        last = len(samples) - 1
        for _ in range(new_n):
            i = int(pos)
            if i >= last:
                resampled.append(samples[-1])
                break
            frac = pos - i
            resampled.append(int(samples[i] + (samples[i + 1] - samples[i]) * frac))
            pos += ratio
        samples = resampled
    return struct.pack("<" + "h" * len(samples), *samples)


def ws_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme in ("ws", "wss") and parsed.path:
        return url
    scheme = "wss" if parsed.scheme in ("https", "wss") else "ws"
    host = parsed.netloc or parsed.path
    return f"{scheme}://{host.rstrip('/')}/v1/worker/duplex"


async def run(args: argparse.Namespace) -> int:
    url = ws_url(args.url)
    pcm = load_wav_pcm16(args.wav)
    chunk_bytes = int(INPUT_RATE * args.chunk_ms / 1000) * 2
    chunks = [pcm[i : i + chunk_bytes] for i in range(0, len(pcm), chunk_bytes) if pcm[i : i + chunk_bytes]]
    t0 = time.perf_counter()
    first_text_ms = None
    first_audio_ms = None
    kinds: list[str] = []
    async with websockets.connect(url, max_size=128 * 1024 * 1024, open_timeout=args.open_timeout) as ws:
        print(f"ws_ready_ms={(time.perf_counter() - t0) * 1000:.1f} url={url}")
        payload = {
            "mode": "full_duplex",
            "system_prompt": args.instructions,
            "config": {"generate_audio": True, "chunk_ms": args.chunk_ms, "sample_rate": INPUT_RATE},
        }
        ref_pcm = load_wav_pcm16(args.ref_wav) if args.ref_wav else pcm
        # Cap Token2Wav ref at 10 s (OpenBMB #11).
        max_ref = INPUT_RATE * 10 * 2
        ref_pcm = ref_pcm[:max_ref]
        ref_b64 = base64.b64encode(pcm16_to_f32le(ref_pcm)).decode("ascii")
        payload["voice"] = {"ref_audio_base64": ref_b64, "tts_ref_audio_base64": ref_b64}
        payload["ref_audio_base64"] = ref_b64
        payload["tts_ref_audio_base64"] = ref_b64
        print(
            "init_fields=voice.ref_audio_base64,voice.tts_ref_audio_base64,"
            "ref_audio_base64,tts_ref_audio_base64 "
            f"ref_samples={len(ref_pcm) // 2}"
        )
        await ws.send(json.dumps({"type": "session.init", "payload": payload}))
        created = False
        sender_task: asyncio.Task | None = None
        stop = asyncio.Event()

        async def sender() -> None:
            silence = b"\x00" * chunk_bytes
            next_send = time.perf_counter()
            idx = 0
            while not stop.is_set():
                piece = chunks[idx] if idx < len(chunks) else silence
                idx += 1
                if idx > len(chunks) and not args.continue_silence:
                    break
                wait = next_send - time.perf_counter()
                if wait > 0:
                    await asyncio.sleep(wait)
                await ws.send(
                    json.dumps(
                        {
                            "type": "input.append",
                            "input": {
                                "audio": base64.b64encode(pcm16_to_f32le(piece)).decode("ascii"),
                                "force_listen": False,
                            },
                        }
                    )
                )
                next_send += args.chunk_ms / 1000.0
                if idx > len(chunks) + int(args.tail_s * 1000 / args.chunk_ms):
                    break

        deadline = time.perf_counter() + args.max_session_s
        while time.perf_counter() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, deadline - time.perf_counter()))
            msg = json.loads(raw)
            typ = str(msg.get("type") or "")
            kinds.append(typ)
            now_ms = (time.perf_counter() - t0) * 1000
            if typ == "session.created" and not created:
                created = True
                print(f"session.created_ms={now_ms:.1f} session_id={msg.get('session_id')}")
                sender_task = asyncio.create_task(sender())
            elif typ == "error":
                print(f"error={json.dumps(msg, ensure_ascii=False)[:400]}", file=sys.stderr)
                return 1
            elif typ == "response.output.delta":
                kind = str(msg.get("kind") or "")
                if kind == "text" and first_text_ms is None and msg.get("text"):
                    first_text_ms = now_ms
                    print(f"first_text_ms={now_ms:.1f} text={msg.get('text')!r}")
                if kind == "audio" and first_audio_ms is None and msg.get("audio"):
                    first_audio_ms = now_ms
                    raw_audio = base64.b64decode(msg["audio"])
                    dur_ms = (len(raw_audio) // 4) / OUTPUT_RATE * 1000
                    print(f"first_audio_ms={now_ms:.1f} duration_ms={dur_ms:.1f}")
                if kind == "listen" and first_audio_ms is not None and args.stop_on_listen:
                    break
            elif typ == "session.closed":
                print(f"session.closed reason={msg.get('reason')}")
                break
        stop.set()
        if sender_task is not None:
            sender_task.cancel()
            try:
                await sender_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            await ws.send(json.dumps({"type": "session.close", "reason": "user_stop"}))
        except Exception:
            pass
    print(f"first_text_ms={first_text_ms}")
    print(f"first_audio_ms={first_audio_ms}")
    print(f"events={','.join(kinds[:32])}")
    return 0 if first_audio_ms is not None else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:22400")
    parser.add_argument("--wav", required=True)
    parser.add_argument("--ref-wav", default="", help="Optional clone WAV. Default: reuse --wav.")
    parser.add_argument("--instructions", default="You are a helpful voice assistant. Reply in English.")
    parser.add_argument("--chunk-ms", type=int, default=1000)
    parser.add_argument("--tail-s", type=float, default=3.0)
    parser.add_argument("--continue-silence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-on-listen", action="store_true", default=True)
    parser.add_argument("--max-session-s", type=float, default=30.0)
    parser.add_argument("--open-timeout", type=float, default=15.0)
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
