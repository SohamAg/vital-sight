from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from src.common.schemas import Detection, EventCandidate, PoseFeat


def _put_text(img, text: str, org: Tuple[int, int], color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_overlays(
    frame_bgr: np.ndarray,
    frame_idx: int,
    fps_est: float,
    detections: List[Detection],
    poses: Dict[str, PoseFeat],
    events: List[EventCandidate],
) -> None:
    h, w = frame_bgr.shape[:2]

    # HUD
    _put_text(
        frame_bgr,
        f"{w}x{h} | frame {frame_idx} | {fps_est:.1f} FPS",
        (10, 20),
        (180, 255, 180),
    )

    # Index events by track for this frame (best-effort)
    evt_by_track: Dict[str, EventCandidate] = {}
    for e in events:
        evt_by_track[e.track_id] = e

    for det in detections:
        x, y, bw, bh = map(int, det.bbox)
        # Box
        cv2.rectangle(frame_bgr, (x, y), (x + bw, y + bh), (255, 255, 255), 1)
        # Label
        _put_text(frame_bgr, f"{det.track_id} {det.conf:.2f}", (x + 2, max(0, y - 6)))

        # Pose details
        pf = poses.get(det.track_id)
        if pf is not None:
            # Torso line: mid-shoulder to mid-hip if we can approximate from angle
            # We cannot reconstruct keypoints here without exposing them; instead show text
            _put_text(
                frame_bgr,
                f"angle:{pf.torso_angle_deg:.1f}°  motion:{pf.motion_energy:.2f}",
                (x + 2, y + 14),
                (200, 220, 255),
            )

        # Event chips
        ev = evt_by_track.get(det.track_id)
        if ev:
            # Draw a small pill with cues and score
            chip_y = y + bh + 16
            chip_x = x
            cues_text = ",".join(ev.cues)
            text = f"{ev.type} {ev.score:.2f}  [{cues_text}]"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            pad = 4
            cv2.rectangle(
                frame_bgr,
                (chip_x - pad, chip_y - th - pad),
                (chip_x + tw + pad, chip_y + pad),
                (60, 50, 180),
                -1,
            )
            _put_text(frame_bgr, text, (chip_x, chip_y), (255, 255, 255))


