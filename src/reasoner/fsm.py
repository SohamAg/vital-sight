from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import numpy as np

from src.common.schemas import Detection, EventCandidate, PoseFeat


@dataclass
class _TrackState:
    last_centroid_y: Tuple[float, float] | None  # (y, ts)
    history: Deque[Tuple[float, float, bool]]  # (ts, motion_energy, is_supine)
    last_score: float
    last_emit_ts: float | None
    last_emit_type: str | None


class TemporalReasoner:
    """Short-memory rule engine to infer collapse/immobility events."""

    def __init__(
        self,
        drop_thr_px: float,
        angle_thr_deg: float,
        motion_thr: float,
        immobile_t: float,
        alert_score_thr: float,
        cam_id: str = "cam-0",
    ):
        self.drop_thr_px = float(drop_thr_px)
        self.angle_thr_deg = float(angle_thr_deg)
        self.motion_thr = float(motion_thr)
        self.immobile_t = float(immobile_t)
        self.alert_score_thr = float(alert_score_thr)
        self.cam_id = cam_id
        self._state: Dict[str, _TrackState] = {}
        self._history_window = max(self.immobile_t, 2.0)

    def _get_state(self, track_id: str) -> _TrackState:
        st = self._state.get(track_id)
        if st is None:
            st = _TrackState(
                last_centroid_y=None,
                history=deque(),
                last_score=0.0,
                last_emit_ts=None,
                last_emit_type=None,
            )
            self._state[track_id] = st
        return st

    def _sudden_drop(self, st: _TrackState, centroid_y: float, ts: float) -> bool:
        """Detect a sudden downward movement within ~0.5 seconds exceeding threshold."""
        sudden = False
        last = st.last_centroid_y
        if last is not None:
            last_y, last_ts = last
            if (ts - last_ts) <= 0.5 and (centroid_y - last_y) > self.drop_thr_px:
                sudden = True
        st.last_centroid_y = (centroid_y, ts)
        return sudden

    def _immobile_seconds(self, st: _TrackState, ts: float) -> float:
        """Longest recent consecutive idle duration where motion_energy < motion_thr."""
        # Ensure history only contains last window
        while st.history and (ts - st.history[0][0]) > self._history_window:
            st.history.popleft()
        # Compute trailing consecutive idle duration ending at latest timestamp
        if not st.history:
            return 0.0
        # Sort by ts just in case
        items = list(st.history)
        items.sort(key=lambda x: x[0])
        # Walk from end backwards until motion exceeds threshold
        end_ts = items[-1][0]
        t = end_ts
        for idx in range(len(items) - 1, -1, -1):
            tsi, mi, _ = items[idx]
            if mi >= self.motion_thr:
                break
            t = tsi
        return max(0.0, end_ts - t)

    def tick(
        self,
        ts: float,
        detections: List[Detection],
        poses: Dict[str, PoseFeat],
    ) -> List[EventCandidate]:
        events: List[EventCandidate] = []
        # Update per-track logic for current tick
        det_by_id: Dict[str, Detection] = {d.track_id: d for d in detections}

        for tid, det in det_by_id.items():
            st = self._get_state(tid)

            # Compute centroid y (pixels)
            x, y, w, h = det.bbox
            centroid_y = float(y + h / 2.0)
            sudden = self._sudden_drop(st, centroid_y, ts)

            pose = poses.get(tid)
            is_supine = False
            if pose is not None:
                is_supine = pose.torso_angle_deg >= self.angle_thr_deg
                st.history.append((ts, pose.motion_energy, is_supine))
            else:
                st.history.append((ts, 0.0, False))

            # Keep only relevant history
            while st.history and (ts - st.history[0][0]) > self._history_window:
                st.history.popleft()

            imm_secs = self._immobile_seconds(st, ts)
            cues: List[str] = []
            score = 0.0
            if sudden:
                score += 0.30
                cues.append("sudden_drop")
            if is_supine:
                score += 0.35
                cues.append("supine_posture")
            if imm_secs >= self.immobile_t:
                score += 0.25
                cues.append(f"immobile_{round(imm_secs, 1)}s")

            # Decide type
            ev_type = "collapse" if sudden else "prolonged_immobility"

            # Debounce: emit on rising edge
            should_emit = score >= self.alert_score_thr and (
                st.last_emit_type != ev_type or (st.last_score < self.alert_score_thr)
            )
            if should_emit:
                t0 = max(ts - 1.0, st.history[0][0] if st.history else ts)
                events.append(
                    EventCandidate(
                        cam_id=self.cam_id,
                        track_id=tid,
                        t0=t0,
                        t1=ts,
                        type=ev_type,  # type: ignore[arg-type]
                        score=min(1.0, round(score, 3)),
                        cues=cues,
                    )
                )
                st.last_emit_ts = ts
                st.last_emit_type = ev_type
            st.last_score = score

        return events


