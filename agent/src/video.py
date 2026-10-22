"""RGB24 LiveKit frames → JPEG stills at VIDEO_FPS."""

from __future__ import annotations

import io
import logging
import os
import time

from PIL import Image

from trace import trace_stream

LOG = logging.getLogger("video")
VIDEO_FPS = float(os.environ.get("VIDEO_FPS", "1"))
# MiniCPM-o duplex takes ~1 JPEG per 1 s audio unit. Keep the long side small
# so vision tokens fit stage-0 max_model_len 4096.
VIDEO_MAX_SIDE = int(os.environ.get("VIDEO_MAX_SIDE", "448"))
MIN_JPEG = 512


def packed_rgb24(raw: bytes, width: int, height: int) -> bytes | None:
    """Drop LiveKit row stride so PIL sees tightly packed RGB24."""
    if width <= 0 or height <= 0 or not raw:
        return None
    row = width * 3
    expected = row * height
    if len(raw) == expected:
        return raw
    if len(raw) < expected:
        return None
    stride = len(raw) // height
    if stride < row or stride * height > len(raw):
        return raw[:expected]
    return b"".join(raw[i * stride : i * stride + row] for i in range(height))


def rgb_to_jpeg(raw: bytes, size: tuple[int, int], quality: int = 70) -> bytes | None:
    w, h = size
    packed = packed_rgb24(raw, w, h)
    if packed is None:
        return None
    try:
        img = Image.frombytes("RGB", (w, h), packed)
        max_side = max(VIDEO_MAX_SIDE, 64)
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        jpeg = buf.getvalue()
    except Exception:
        LOG.debug("jpeg encode failed", exc_info=True)
        return None
    if len(jpeg) < MIN_JPEG or jpeg[:2] != b"\xff\xd8":
        return None
    return jpeg


class FrameSampler:
    """Keep the newest JPEG and emit at most VIDEO_FPS stills per second."""

    def __init__(self, fps: float = VIDEO_FPS) -> None:
        self.period = 1.0 / max(fps, 0.1)
        self._last = 0.0
        self._jpeg: bytes | None = None
        self._size = (0, 0)

    def offer(self, raw: bytes, size: tuple[int, int], fmt: str = "RGB") -> bytes | None:
        if fmt.upper() not in {"RGB", "RGB24"}:
            return None
        jpeg = rgb_to_jpeg(raw, size)
        if not jpeg:
            return None
        self._jpeg = jpeg
        self._size = size
        now = time.monotonic()
        if now - self._last < self.period:
            return None
        self._last = now
        trace_stream("speech", "video.frame", bytes=len(jpeg), w=size[0], h=size[1])
        return jpeg

    @property
    def latest(self) -> bytes | None:
        return self._jpeg
