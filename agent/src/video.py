"""RGB24 LiveKit frames → JPEG stills at VIDEO_FPS."""

from __future__ import annotations

import io
import logging
import os
import time

from PIL import Image

from trace import trace, trace_dbg

LOG = logging.getLogger("video")
VIDEO_FPS = float(os.environ.get("VIDEO_FPS", "1"))
MIN_JPEG = 2048


def rgb_to_jpeg(raw: bytes, size: tuple[int, int], quality: int = 70) -> bytes | None:
    w, h = size
    if w <= 0 or h <= 0 or not raw:
        return None
    expected = w * h * 3
    if len(raw) < expected:
        return None
    try:
        img = Image.frombytes("RGB", (w, h), raw[:expected])
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        jpeg = buf.getvalue()
    except Exception:
        LOG.debug("jpeg encode failed", exc_info=True)
        return None
    if len(jpeg) < MIN_JPEG:
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
            trace_dbg("speech", "video.chunk", bytes=len(jpeg), w=size[0], h=size[1], dropped=True)
            return None
        self._last = now
        trace("speech", "video.frame", bytes=len(jpeg), w=size[0], h=size[1])
        return jpeg

    @property
    def latest(self) -> bytes | None:
        return self._jpeg
