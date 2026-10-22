"""Process-wide logging. Pin noisy WebRTC libraries so SA_TRACE stays visible."""

from __future__ import annotations

import logging
import os


class _SkipHealth(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return "GET /health" not in msg


def configure_logging() -> None:
    level_name = (os.environ.get("LOG_LEVEL") or "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    logging.getLogger("uvicorn.access").addFilter(_SkipHealth())
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    for name in ("livekit", "livekit.rtc", "aiortc", "aioice"):
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("sa.trace").setLevel(logging.INFO)
