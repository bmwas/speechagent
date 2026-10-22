"""RGB24 LiveKit frames → JPEG at a per-engine FPS.

MiniCPM duplex attaches ~1 JPEG per 1 s audio unit (``VIDEO_FPS``).
Qwen2.5-Omni live video-stream wants ``QWEN_VIDEO_FPS`` (default 5) so the
server session buffer is a real video window, not one still.
"""

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


def rgb_to_jpeg(
    raw: bytes,
    size: tuple[int, int],
    quality: int = 70,
    max_side: int | None = None,
) -> bytes | None:
    w, h = size
    packed = packed_rgb24(raw, w, h)
    if packed is None:
        return None
    try:
        img = Image.frombytes("RGB", (w, h), packed)
        cap = max(max_side if max_side is not None else VIDEO_MAX_SIDE, 64)
        if max(img.size) > cap:
            img.thumbnail((cap, cap), Image.Resampling.LANCZOS)
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
    """Emit at most ``fps`` JPEGs per second from the current LiveKit frame."""

    def __init__(self, fps: float = VIDEO_FPS, max_side: int | None = None) -> None:
        self.period = 1.0 / max(fps, 0.1)
        self.max_side = VIDEO_MAX_SIDE if max_side is None else int(max_side)
        self._last = 0.0
        self._jpeg: bytes | None = None
        self._size = (0, 0)

    def offer(self, raw: bytes, size: tuple[int, int], fmt: str = "RGB") -> bytes | None:
        if fmt.upper() not in {"RGB", "RGB24"}:
            return None
        now = time.monotonic()
        if now - self._last < self.period:
            return None
        jpeg = rgb_to_jpeg(raw, size, max_side=self.max_side)
        if not jpeg:
            return None
        self._jpeg = jpeg
        self._size = size
        self._last = now
        trace_stream("speech", "video.frame", bytes=len(jpeg), w=size[0], h=size[1], fps=round(1.0 / self.period, 2))
        return jpeg

    @property
    def latest(self) -> bytes | None:
        return self._jpeg
