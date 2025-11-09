from __future__ import annotations

from typing import Literal, Optional, Tuple

import cv2
import numpy as np
import time

from src.common.utils import now_s


class VideoSource:
    """Unified webcam/file reader with resize and FPS normalization via frame skipping.

    read() returns (ok, frame_bgr, ts_monotonic)
    """

    def __init__(
        self,
        kind: Literal["webcam", "file"],
        path: Optional[str],
        target_w: int,
        target_h: int,
        target_fps: int,
    ):
        self.kind = kind
        self.path = path
        self.target_w = int(target_w)
        self.target_h = int(target_h)
        self.target_fps = max(1, int(target_fps))

        if kind == "webcam":
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            if not path:
                raise ValueError("File source requires a valid --path")
            self.cap = cv2.VideoCapture(path)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {kind} {path or ''}")

        self._last_emit_ts = now_s()
        self._min_dt = 1.0 / float(self.target_fps)
        self._frame_idx = 0

        # For file sources, we will read every frame and downsample by time emit cadence.
        # For webcam, we poll and emit frames at approximately target_fps using sleep.

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[1] == self.target_w and frame.shape[0] == self.target_h:
            return frame
        return cv2.resize(frame, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)

    def read(self) -> Tuple[bool, np.ndarray, float]:
        # Enforce timing to approximate target FPS
        now = now_s()
        dt = now - self._last_emit_ts
        if dt < self._min_dt:
            time.sleep(max(0.0, self._min_dt - dt))

        ok, frame = self.cap.read()
        if not ok:
            return False, np.zeros((1, 1, 3), dtype=np.uint8), now_s()

        frame = self._resize(frame)
        ts = now_s()
        self._last_emit_ts = ts
        self._frame_idx += 1
        return True, frame, ts

    def release(self) -> None:
        try:
            if hasattr(self, "cap") and self.cap is not None:
                self.cap.release()
        except Exception:
            pass


