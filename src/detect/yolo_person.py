from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from ultralytics import YOLO

from src.common.schemas import Detection


def _iou_xywh(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class _Track:
    bbox: Tuple[float, float, float, float]
    track_id: str
    last_seen: int


class YoloPersonDetector:
    """Ultralytics YOLO wrapper for person detection with lightweight ID association."""

    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.4):
        self.model = YOLO(model_name)
        self.conf = float(conf)
        self._tracks: Dict[str, _Track] = {}
        self._frame_counter = 0
        self._id_counter = 0
        self._max_age = 30  # frames to keep unmatched tracks

    def _assign_ids(
        self, boxes_xywh: List[Tuple[float, float, float, float]]
    ) -> List[str]:
        assigned_ids: List[str] = [""] * len(boxes_xywh)
        used_tracks: set[str] = set()

        # Try greedy matching by IoU threshold, then nearest center distance
        for i, box in enumerate(boxes_xywh):
            best_id = None
            best_iou = 0.0
            for tid, tr in self._tracks.items():
                if tid in used_tracks:
                    continue
                iou = _iou_xywh(box, tr.bbox)
                if iou > 0.3 and iou > best_iou:
                    best_iou = iou
                    best_id = tid
            if best_id is None:
                # fallback: nearest centroid
                bx, by, bw, bh = box
                bc = np.array([bx + bw / 2.0, by + bh / 2.0])
                min_dist = float("inf")
                for tid, tr in self._tracks.items():
                    if tid in used_tracks:
                        continue
                    tx, ty, tw, th = tr.bbox
                    tc = np.array([tx + tw / 2.0, ty + th / 2.0])
                    dist = float(np.linalg.norm(bc - tc))
                    if dist < min_dist and dist < 0.5 * max(bw + bh, tw + th):
                        min_dist = dist
                        best_id = tid

            if best_id is None:
                best_id = f"p-{self._id_counter}"
                self._id_counter += 1

            assigned_ids[i] = best_id
            used_tracks.add(best_id)

        # Update track store
        new_tracks: Dict[str, _Track] = {}
        for i, box in enumerate(boxes_xywh):
            tid = assigned_ids[i]
            new_tracks[tid] = _Track(bbox=box, track_id=tid, last_seen=self._frame_counter)

        # Keep old tracks that are not too old
        for tid, tr in self._tracks.items():
            if tid not in new_tracks and (self._frame_counter - tr.last_seen) <= self._max_age:
                new_tracks[tid] = tr

        self._tracks = new_tracks
        return assigned_ids

    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Run YOLO, filter to person class, return list of Detection."""
        self._frame_counter += 1
        h, w = frame_bgr.shape[:2]
        results = self.model.predict(source=frame_bgr, conf=self.conf, verbose=False)
        if not results:
            return []
        boxes_xywh: List[Tuple[float, float, float, float]] = []
        confs: List[float] = []
        for r in results:
            if not hasattr(r, "boxes") or r.boxes is None:
                continue
            for b in r.boxes:
                cls_id = int(b.cls.item()) if b.cls is not None else -1
                if cls_id != 0:  # 0 == person
                    continue
                xyxy = b.xyxy.cpu().numpy().reshape(-1)
                x1, y1, x2, y2 = map(float, xyxy[:4])
                x = max(0.0, min(x1, w - 1.0))
                y = max(0.0, min(y1, h - 1.0))
                bw = max(0.0, min(x2 - x, w - x))
                bh = max(0.0, min(y2 - y, h - y))
                boxes_xywh.append((x, y, bw, bh))
                confs.append(float(b.conf.item() if b.conf is not None else 0.0))

        if not boxes_xywh:
            return []

        track_ids = self._assign_ids(boxes_xywh)
        detections: List[Detection] = []
        for (x, y, bw, bh), conf, tid in zip(boxes_xywh, confs, track_ids):
            detections.append(Detection(track_id=tid, bbox=[x, y, bw, bh], conf=conf))
        return detections


