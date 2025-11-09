from __future__ import annotations

import json
import time
from typing import Iterable, Optional, Tuple

import cv2

from .schemas import EventCandidate


def now_s() -> float:
    """Monotonic seconds for timestamps."""
    return time.monotonic()


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def clamp_bbox(bbox: Iterable[float], width: int, height: int) -> Tuple[int, int, int, int]:
    """Clamp [x,y,w,h] bbox within 0..width/height and return ints."""
    x, y, w, h = [float(v) for v in bbox]
    x = clamp(x, 0.0, float(width - 1))
    y = clamp(y, 0.0, float(height - 1))
    w = clamp(w, 0.0, float(width - x))
    h = clamp(h, 0.0, float(height - y))
    return int(x), int(y), int(w), int(h)


def format_event_json(event: EventCandidate) -> str:
    """Compact one-line JSON suitable for stdout."""
    def round_if_float(v):
        if isinstance(v, float):
            return round(v, 3)
        if isinstance(v, list):
            return [round_if_float(x) for x in v]
        return v

    data = event.model_dump()
    data = {k: round_if_float(v) for k, v in data.items()}
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def make_video_writer(path: str, width: int, height: int, fps: int):
    """Create a cross-platform MP4 writer. Returns cv2.VideoWriter or None on failure."""
    # Prefer mp4v for Windows/macOS. If unavailable, try avc1.
    for fourcc_str in ("mp4v", "avc1", "H264", "XVID"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
    return None


