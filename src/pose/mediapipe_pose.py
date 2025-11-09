from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from math import atan2, degrees
import mediapipe as mp

from src.common.schemas import Detection, PoseFeat
from src.common.utils import clamp_bbox


_POSE = mp.solutions.pose


class PoseEstimator:
    """MediaPipe Pose wrapper computing torso angle and motion energy per track."""

    def __init__(self):
        self.pose = _POSE.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # track_id -> (landmarks in pixel coords, crop_size)
        self._prev_landmarks: Dict[str, Tuple[np.ndarray, Tuple[int, int]]] = {}

    def _compute_torso_angle_deg(self, landmarks_xy: np.ndarray) -> float:
        # Use mid-shoulder (11,12) and mid-hip (23,24)
        # landmarks_xy: shape (N,2) in pixel coords, mediapipe landmark indexing
        # If any of required landmarks missing, return 0.0
        try:
            shoulder_l = landmarks_xy[11]
            shoulder_r = landmarks_xy[12]
            hip_l = landmarks_xy[23]
            hip_r = landmarks_xy[24]
        except Exception:
            return 0.0
        shoulder_mid = 0.5 * (shoulder_l + shoulder_r)
        hip_mid = 0.5 * (hip_l + hip_r)
        vec = hip_mid - shoulder_mid
        # Angle to vertical axis (y down). Vertical vector is (0, 1).
        # We compute angle between vec and vertical as atan2(|dx|, |dy|)
        dx, dy = float(vec[0]), float(vec[1])
        ang = degrees(atan2(abs(dx), abs(dy) + 1e-6))
        # Clamp into [0, 180]
        return float(max(0.0, min(180.0, ang)))

    def _motion_energy(
        self,
        track_id: str,
        landmarks_xy: np.ndarray,
        crop_size: Tuple[int, int],
    ) -> float:
        prev = self._prev_landmarks.get(track_id)
        if prev is None:
            return 0.0
        prev_xy, prev_size = prev
        # If crop size changed significantly, reset energy
        if prev_size != crop_size or prev_xy.shape != landmarks_xy.shape:
            return 0.0
        diffs = np.linalg.norm(landmarks_xy - prev_xy, axis=1)
        norm = max(1.0, max(crop_size))
        return float(np.mean(diffs) / norm)

    def infer_one(self, frame_bgr: np.ndarray, det: Detection) -> Optional[PoseFeat]:
        x, y, w, h = clamp_bbox(det.bbox, frame_bgr.shape[1], frame_bgr.shape[0])
        if w <= 1 or h <= 1:
            return None
        crop = frame_bgr[y : y + h, x : x + w]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        if not result.pose_landmarks or not result.pose_landmarks.landmark:
            return None
        lm = result.pose_landmarks.landmark
        # Convert normalized landmarks to pixel coordinates within crop
        pts = []
        for p in lm:
            px = float(p.x) * float(w)
            py = float(p.y) * float(h)
            pts.append([px, py])
        landmarks_xy = np.array(pts, dtype=np.float32)  # shape (33,2)
        angle = self._compute_torso_angle_deg(landmarks_xy)
        energy = self._motion_energy(det.track_id, landmarks_xy, (w, h))
        # cache current
        self._prev_landmarks[det.track_id] = (landmarks_xy, (w, h))
        return PoseFeat(track_id=det.track_id, torso_angle_deg=angle, motion_energy=energy)


