from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel, Field


class Detection(BaseModel):
    """Person detection produced by the YOLO wrapper.

    - bbox: [x, y, w, h] in pixels in the resized working frame (top-left origin).
    - track_id: stable id like "p-3" within the current run.
    - conf: detector confidence in [0, 1].
    """

    track_id: str = Field(description="Stable id like 'p-3'")
    bbox: List[float] = Field(min_items=4, max_items=4, description="[x,y,w,h] in pixels")
    conf: float = Field(ge=0.0, le=1.0)


class PoseFeat(BaseModel):
    """Pose-derived features for temporal reasoning."""

    track_id: str
    torso_angle_deg: float = Field(
        ge=0.0, le=180.0, description="0=vertical, 90=horizontal; angle to vertical"
    )
    motion_energy: float = Field(
        ge=0.0, description="Mean L2 delta of landmarks normalized by crop size"
    )


class EventCandidate(BaseModel):
    """Temporal reasoning output for a single track over an interval [t0, t1]."""

    cam_id: str
    track_id: str
    t0: float
    t1: float
    type: Literal["collapse", "prolonged_immobility", "assist_needed"]
    score: float = Field(ge=0.0, le=1.0)
    cues: List[str] = Field(description="e.g. ['sudden_drop', 'supine_posture', 'immobile_8.2s']")


